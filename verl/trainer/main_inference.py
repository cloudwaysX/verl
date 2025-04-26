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
def process_batch(model_path, batch_indices, batch_texts, max_length, pooling_method, normalize):
    import ray as _ray
    import os as _os
    import torch as _torch
    import numpy as _np
    from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel

    # Restrict to the GPU assigned by Ray
    gpu_ids = _ray.get_gpu_ids()
    assert len(gpu_ids) == 1, f"Expected exactly one GPU per worker, got: {gpu_ids}"
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    device = _torch.device("cuda:0")

    tokenizer = _AutoTokenizer.from_pretrained(model_path)
    model = _AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Tokenize and move to device
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

    # Return original indices and embeddings together
    return batch_indices, embeddings

@hydra.main(config_path='config', config_name='inference', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Ensure driver sees all GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(config.embedding.n_gpus_per_node))
    if not ray.is_initialized():
        ray.init()

    # Load model locally
    model_name = config.embedding.model.name
    model_path = copy_to_local(model_name)

    # Read and optionally subsample data
    df = pd.read_parquet(config.data.path)
    if config.data.train_ratio < 1:
        n = int(len(df) * config.data.train_ratio)
        if config.data.train_ratio_seed is not None:
            np.random.seed(config.data.train_ratio_seed)
            df = df.sample(frac=1, random_state=config.data.train_ratio_seed).reset_index(drop=True)
        df = df.head(n)

    # Prepare prompts and indices
    raw_prompts = []
    for prompt in df[config.data.prompt_key].tolist():
        if hasattr(prompt, '__iter__') and not isinstance(prompt, str):
            first = prompt[0]
            raw_prompts.append(first['content'] if isinstance(first, dict) and 'content' in first else str(first))
        else:
            raw_prompts.append(prompt)
    total = len(raw_prompts)
    all_indices = list(range(total))

    # Batch indices and texts
    batch_size = config.embedding.batch_size
    batches_idx = [all_indices[i:i + batch_size] for i in range(0, total, batch_size)]
    batches_txt = [[raw_prompts[i] for i in idxs] for idxs in batches_idx]

    # Launch Ray tasks with index binding
    futures = [
        process_batch.remote(
            model_path,
            idxs,
            texts,
            config.embedding.max_length,
            config.embedding.pooling_method,
            config.embedding.normalize_embeddings
        )
        for idxs, texts in zip(batches_idx, batches_txt)
    ]

    # Gather all results
    results = ray.get(futures)

    # Allocate and fill embeddings by original index
    # Determine dimension
    emb_dim = results[0][1].shape[1]
    embeddings = np.zeros((total, emb_dim), dtype=results[0][1].dtype)
    for idxs, emb in results:
        print(f"Processing batch {idxs}")
        embeddings[idxs, :] = emb

    # Save
    makedirs(config.embedding.output_path, exist_ok=True)
    out_file = os.path.join(config.embedding.output_path, 'embeddings.npy')
    print(f"Saving embeddings to {out_file}")
    print(f"Embedding shape: {embeddings.shape}")
    np.save(out_file, embeddings)
    print("Done!")

if __name__ == '__main__':
    main()
