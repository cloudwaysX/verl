import ray
import numpy as np
import hydra
import os
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs

import pandas as pd

@ray.remote(num_gpus=1)
def process_batch(
    model_name,
    framework,
    batch_indices,
    batch_texts,
    max_length,
    pooling_method,
    normalize,
    prompt_name=None
):
    """
    Worker function: encodes a batch of texts using either HuggingFace AutoModel (decoder-based)
    or SentenceTransformer (encoder-based).
    """
    import ray as _ray
    import os as _os
    import numpy as _np
    import torch as _torch

    # Pin to assigned GPU
    gpu_ids = _ray.get_gpu_ids()
    assert len(gpu_ids) == 1, f"Expected one GPU, got {gpu_ids}"
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    device = _torch.device("cuda:0")

    if framework == 'hf':
        # HuggingFace decoder-style embedding
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device).eval()

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        with _torch.no_grad():
            out = model(**encoded)

        if pooling_method == 'cls':
            embs = out.last_hidden_state[:, 0, :].cpu().numpy()
        elif pooling_method == 'mean':
            tokens = out.last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).expand(tokens.size()).float()
            summed = (_torch.sum(tokens * mask, dim=1) / _torch.clamp(mask.sum(1), min=1e-9)).cpu().numpy()
            embs = summed
        else:
            raise ValueError(f"Unknown pooling: {pooling_method}")
        if normalize:
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    elif framework == 'st':
        # SentenceTransformer encoder-style
        model = SentenceTransformer(model_name)
        if max_length is not None:
            model.max_seq_length = max_length
        if prompt_name:
            embs = model.encode(batch_texts, prompt_name=prompt_name)
        else:
            embs = model.encode(batch_texts)
        if isinstance(embs, _torch.Tensor):
            embs = embs.cpu().numpy()
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    return batch_indices, embs

@hydra.main(config_path='config', config_name='inference', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Expose GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(config.embedding.n_gpus_per_node))
    if not ray.is_initialized():
        ray.init()

    # Determine framework
    framework = config.embedding.get('framework', 'hf')  # 'hf' or 'st'
    model_name = config.embedding.model.name
    if framework == 'hf':
        model_path = copy_to_local(model_name)
    else:
        model_path = model_name  # SBERT caches internally

    # Load dataset
    df = pd.read_parquet(config.data.path)
    if config.data.train_ratio < 1:
        n = int(len(df) * config.data.train_ratio)
        if config.data.train_ratio_seed is not None:
            np.random.seed(config.data.train_ratio_seed)
            df = df.sample(frac=1, random_state=config.data.train_ratio_seed).reset_index(drop=True)
        df = df.head(n)

    # Extract texts
    texts = []
    for x in df[config.data.prompt_key].tolist():
        if hasattr(x, '__iter__') and not isinstance(x, str):
            elem = x[0]
            texts.append(elem['content'] if isinstance(elem, dict) and 'content' in elem else str(elem))
        else:
            texts.append(x)

    total = len(texts)
    indices = list(range(total))

    # Build batches
    bs = config.embedding.batch_size
    batches_idx = [indices[i:i+bs] for i in range(0, total, bs)]
    batches_txt = [[texts[i] for i in batch] for batch in batches_idx]

    # Launch tasks
    prompt_name = config.embedding.get('prompt_name', None)
    futures = [
        process_batch.remote(
            model_path,
            framework,
            idxs,
            txts,
            config.embedding.max_length,
            config.embedding.pooling_method,
            config.embedding.normalize_embeddings,
            prompt_name
        )
        for idxs, txts in zip(batches_idx, batches_txt)
    ]

    # Collect
    results = ray.get(futures)
    dim = results[0][1].shape[1]
    embeddings = np.zeros((total, dim), dtype=results[0][1].dtype)
    for idxs, emb in results:
        embeddings[idxs, :] = emb

    # Save
    makedirs(config.embedding.output_path, exist_ok=True)
    out = os.path.join(config.embedding.output_path, 'embeddings.npy')
    print(f"Saving to {out}, shape {embeddings.shape}")
    np.save(out, embeddings)
    print("Done.")

if __name__ == '__main__':
    main()
