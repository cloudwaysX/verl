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
def process_batch(model_path, batch_texts, max_length, pooling_method, normalize):
    # Dynamically obtain and use the GPU assigned by Ray
    import ray as _ray
    import os as _os
    import torch as _torch
    import numpy as _np
    from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel

    gpu_ids = _ray.get_gpu_ids()
    assert len(gpu_ids) == 1, f"Expected exactly one GPU per worker, got: {gpu_ids}"
    # Restrict this process to the assigned GPU
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    device = _torch.device("cuda:0")
    print(f"Ray assigned GPU id: {gpu_ids[0]}, using device: {device}")

    tokenizer = _AutoTokenizer.from_pretrained(model_path)
    model = _AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    encoded_input = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(device)

    with _torch.no_grad():
        model_output = model(**encoded_input)

    if pooling_method == 'cls':
        batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
    elif pooling_method == 'mean':
        token_embeddings = model_output[0]
        mask = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = _torch.sum(token_embeddings * mask, dim=1)
        counts = _torch.clamp(mask.sum(dim=1), min=1e-9)
        batch_embeddings = (summed / counts).cpu().numpy()
    else:
        raise NotImplementedError(f"Pooling method {pooling_method} not implemented.")

    if normalize:
        batch_embeddings = batch_embeddings / _np.linalg.norm(batch_embeddings, axis=1, keepdims=True)

    return batch_embeddings

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

    model_name = config.embedding.model.name
    model_path = copy_to_local(model_name)

    # Load and optionally sample dataset
    dataset = pd.read_parquet(config.data.path)
    if config.data.train_ratio < 1:
        size = int(len(dataset) * config.data.train_ratio)
        if config.data.train_ratio_seed is not None:
            np.random.seed(config.data.train_ratio_seed)
            dataset = dataset.sample(frac=1, random_state=config.data.train_ratio_seed).reset_index(drop=True)
        dataset = dataset.head(size)

    texts = []
    for prompt in dataset[config.data.prompt_key].tolist():
        if hasattr(prompt, '__iter__') and not isinstance(prompt, str):
            first = prompt[0]
            texts.append(first['content'] if isinstance(first, dict) and 'content' in first else str(first))
        else:
            texts.append(prompt)

    # Batch and distribute
    batch_size = config.embedding.batch_size
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    futures = [
        process_batch.remote(
            model_path,
            batch,
            config.embedding.max_length,
            config.embedding.pooling_method,
            config.embedding.normalize_embeddings
        )
        for batch in batches
    ]

    # Collect results
    embeddings = []
    total = len(texts)
    with tqdm(total=total) as pbar:
        for batch, fut in zip(batches, futures):
            out = ray.get(fut)
            embeddings.extend(out)
            pbar.update(len(batch))

    embeddings = np.array(embeddings)

    # Save
    makedirs(config.embedding.output_path, exist_ok=True)
    out_file = os.path.join(config.embedding.output_path, 'embeddings.npy')
    print(f"Saving embeddings to {out_file}")
    print(f"Embedding shape: {embeddings.shape}")
    np.save(out_file, embeddings)
    print("Done!")

if __name__ == '__main__':
    main()
