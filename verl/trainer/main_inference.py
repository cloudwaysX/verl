import ray
import numpy as np
import hydra
import os
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs

import pandas as pd

@ray.remote(num_gpus=1)
def process_batch(model_path, batch_indices, batch_texts, max_length, pooling_method, normalize, encode_task=None):
    import ray as _ray
    import os as _os
    import torch as _torch
    import numpy as _np
    from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel

    gpu_ids = _ray.get_gpu_ids()
    assert len(gpu_ids) == 1, f"Expected exactly one GPU per worker, got: {gpu_ids}"
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    device = _torch.device("cuda:0")

    tokenizer = _AutoTokenizer.from_pretrained(model_path)
    model = _AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()

    # If the model implements an .encode(...) API, use it directly
    if hasattr(model, 'encode'):
        # encode returns NumPy or Torch tensor; convert to NumPy
        if encode_task:
            embeddings = model.encode(batch_texts, task=encode_task)
        else:
            embeddings = model.encode(batch_texts)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
    else:
        # Fallback to manual pooling via tokenizer + forward pass
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        with _torch.no_grad():
            output = model(**encoded)

        if pooling_method == 'cls':
            embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        elif pooling_method == 'mean':
            tokens = output[0]
            mask = encoded['attention_mask'].unsqueeze(-1).expand(tokens.size()).float()
            summed = (_torch.sum(tokens * mask, dim=1) / _torch.clamp(mask.sum(dim=1), min=1e-9)).cpu().numpy()
            embeddings = summed
        else:
            raise NotImplementedError(f"Pooling method {pooling_method} not implemented.")
        if normalize:
            embeddings = embeddings / _np.linalg.norm(embeddings, axis=1, keepdims=True)

    return batch_indices, embeddings

@hydra.main(config_path='config', config_name='inference', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(config.embedding.n_gpus_per_node))
    if not ray.is_initialized():
        ray.init()

    model_name = config.embedding.model.name
    model_path = copy_to_local(model_name)

    df = pd.read_parquet(config.data.path)
    if config.data.train_ratio < 1:
        n = int(len(df) * config.data.train_ratio)
        if config.data.train_ratio_seed is not None:
            np.random.seed(config.data.train_ratio_seed)
            df = df.sample(frac=1, random_state=config.data.train_ratio_seed).reset_index(drop=True)
        df = df.head(n)

    raw_prompts = []
    for prompt in df[config.data.prompt_key].tolist():
        if hasattr(prompt, '__iter__') and not isinstance(prompt, str):
            first = prompt[0]
            raw_prompts.append(first['content'] if isinstance(first, dict) and 'content' in first else str(first))
        else:
            raw_prompts.append(prompt)
    total = len(raw_prompts)
    all_indices = list(range(total))

    batch_size = config.embedding.batch_size
    batches_idx = [all_indices[i:i + batch_size] for i in range(0, total, batch_size)]
    batches_txt = [[raw_prompts[i] for i in idxs] for idxs in batches_idx]

    encode_task = getattr(config.embedding, 'encode_task', None)
    futures = [
        process_batch.remote(
            model_path,
            idxs,
            texts,
            config.embedding.max_length,
            config.embedding.pooling_method,
            config.embedding.normalize_embeddings,
            encode_task
        )
        for idxs, texts in zip(batches_idx, batches_txt)
    ]

    results = ray.get(futures)

    emb_dim = results[0][1].shape[1]
    embeddings = np.zeros((total, emb_dim), dtype=results[0][1].dtype)

    for idxs, emb in results:
        print(f"Processing batch {idxs}")
        embeddings[idxs, :] = emb
        
    print("Example:")
    print(embeddings[0:3])

    # Save
    makedirs(config.embedding.output_path, exist_ok=True)
    out_file = os.path.join(config.embedding.output_path, 'embeddings.npy')
    print(f"Saving embeddings to {out_file}")
    print(f"Embedding shape: {embeddings.shape}")
    np.save(out_file, embeddings)
    print("Done!")
    
    # === Additional distribution statistics ===
    # Per-sample variance
    var_per_sample = np.var(embeddings, axis=1)
    print(f"Sample-wise embedding variance — mean: {var_per_sample.mean():.4f}, std: {var_per_sample.std():.4f}, min: {var_per_sample.min():.4f}, max: {var_per_sample.max():.4f}")

    # Embedding norm distribution
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms — mean: {norms.mean():.4f}, std: {norms.std():.4f}, min: {norms.min():.4f}, max: {norms.max():.4f}")

    # Dimension-wise statistics
    dim_means = embeddings.mean(axis=0)
    dim_vars = embeddings.var(axis=0)
    print(f"Dimension-level mean of means: {dim_means.mean():.4f}, std of means: {dim_means.std():.4f}")
    print(f"Dimension-level mean of variances: {dim_vars.mean():.4f}, std of variances: {dim_vars.std():.4f}")

    # Optionally: count dimensions with high variance
    thresh = dim_vars.mean()
    high_var_dims = np.sum(dim_vars > thresh)
    print(f"Number of dimensions with variance > mean variance ({thresh:.4f}): {high_var_dims}/{emb_dim}")

if __name__ == '__main__':
    main()
