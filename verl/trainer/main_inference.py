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
    # Dynamically get the GPU assigned by Ray
    gpu_ids = ray.get_gpu_ids()
    assert len(gpu_ids) == 1, f"Expected 1 GPU per worker, got: {gpu_ids}"
    gpu_id = gpu_ids[0]
    device = f"cuda:{gpu_id}"

    print(f"Ray assigned GPU id: {gpu_id}, device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    encoded_input = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    if pooling_method == 'cls':
        batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
    elif pooling_method == 'mean':
        token_embeddings = model_output[0]
        input_mask_expanded = encoded_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = batch_embeddings.cpu().numpy()
    else:
        raise NotImplementedError(f"Pooling method {pooling_method} not implemented.")

    if normalize:
        batch_embeddings /= np.linalg.norm(batch_embeddings, axis=1, keepdims=True)

    return batch_embeddings

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

    dataset = pd.read_parquet(config.data.path)
    if config.data.train_ratio < 1:
        size = int(len(dataset) * config.data.train_ratio)
        if config.data.train_ratio_seed is not None:
            np.random.seed(config.data.train_ratio_seed)
            dataset = dataset.sample(frac=1, random_state=config.data.train_ratio_seed).reset_index(drop=True)
        dataset = dataset.head(size)

    prompts_list = dataset[config.data.prompt_key].tolist()

    chat_lst = []
    for prompt in prompts_list:
        if hasattr(prompt, '__iter__') and not isinstance(prompt, str):
            prompt_list = list(prompt)
            if len(prompt) > 0 and isinstance(prompt[0], dict) and 'content' in prompt[0]:
                chat_lst.append(prompt[0]['content'])
            else:
                chat_lst.append(str(prompt[0]) if len(prompt) > 0 else "")
        else:
            chat_lst.append(prompt)

    batch_size = config.embedding.batch_size
    batches = [chat_lst[i:i + batch_size] for i in range(0, len(chat_lst), batch_size)]

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

    embeddings = []
    total_samples = len(chat_lst)

    with tqdm(total=total_samples) as pbar:
        for batch, future in zip(batches, futures):
            batch_embeddings = ray.get(future)
            embeddings.extend(batch_embeddings)
            pbar.update(len(batch))

    embeddings = np.array(embeddings)

    output_dir = config.embedding.output_path
    makedirs(output_dir, exist_ok=True)

    embedding_file = os.path.join(output_dir, "embeddings.npy")
    print(f"Saving embeddings to {embedding_file}")
    print(f"Embedding shape: {embeddings.shape}")
    np.save(embedding_file, embeddings)

    print("Done!")

if __name__ == '__main__':
    main()
