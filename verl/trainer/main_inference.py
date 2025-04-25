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
def process_batch(model_path, batch_texts, device_id, max_length, pooling_method, normalize):
    # Set the device
    device = f"cuda:{device_id}"
    
    # Initialize tokenizer and model for this worker
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Tokenize the batch
    encoded_input = tokenizer(
        batch_texts, 
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(device)
    
    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Get embeddings based on pooling method
    if pooling_method == 'cls':
        batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
    elif pooling_method == 'mean':
        # Mean pooling
        token_embeddings = model_output[0]
        input_mask_expanded = encoded_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = batch_embeddings.cpu().numpy()
    else:
        raise NotImplementedError(f"Pooling method {pooling_method} not implemented.")
    
    # Normalize if required
    if normalize:
        batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
    
    return batch_embeddings

@hydra.main(config_path='config', config_name='inference', version_base=None)
def main(config):
    # Print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Get model information
    model_name = config.embedding.model.name    
    # Download the checkpoint if needed
    model_path = copy_to_local(model_name)
    
    # Read dataset from parquet file
    dataset = pd.read_parquet(config.data.path)
    if config.data.train_ratio < 1:
        size = int(len(dataset) * config.data.train_ratio)
        if config.data.train_ratio_seed is not None:
            np.random.seed(config.data.train_ratio_seed)
            dataset = dataset.sample(frac=1, random_state=config.data.train_ratio_seed).reset_index(drop=True)
        dataset = dataset.head(size)
    
    # Extract prompts from the dataset
    prompts_list = dataset[config.data.prompt_key].tolist()
    
    # Convert list format to simple text if needed
    chat_lst = []
    for prompt in prompts_list:
        if hasattr(prompt, '__iter__') and not isinstance(prompt, str):
            # Convert to a list for consistent handling
            prompt_list = list(prompt)
            # Process chat format
            if len(prompt) > 0 and isinstance(prompt[0], dict) and 'content' in prompt[0]:
                chat_lst.append(prompt[0]['content'])
            else:
                chat_lst.append(str(prompt[0]) if len(prompt) > 0 else "")
        else:
            # Simple text format
            chat_lst.append(prompt)
    
    # Use all 8 GPUs as specified in config
    num_gpus = config.embedding.n_gpus_per_node
    print(f"Using {num_gpus} GPUs for parallel processing")
    
    # Process in batches
    print("Preparing batches for distribution across GPUs...")
    batch_size = config.embedding.batch_size
    
    # Prepare all batches
    all_batches = []
    for i in range(0, len(chat_lst), batch_size):
        batch_texts = chat_lst[i:i+batch_size]
        all_batches.append(batch_texts)
    
    # Distribute batches across GPUs
    gpu_batches = [[] for _ in range(num_gpus)]
    for i, batch in enumerate(all_batches):
        gpu_id = i % num_gpus
        gpu_batches[gpu_id].append(batch)
    
    # Submit tasks to Ray
    print("Submitting tasks to Ray workers...")
    futures = []
    for gpu_id in range(num_gpus):
        for batch in gpu_batches[gpu_id]:
            future = process_batch.remote(
                model_path, 
                batch,
                gpu_id,
                config.embedding.max_length,
                config.embedding.pooling_method,
                config.embedding.normalize_embeddings
            )
            futures.append((len(batch), future))
    
    # Collect results
    print("Computing embeddings on all GPUs...")
    embeddings = []
    total_samples = sum(count for count, _ in futures)
    
    with tqdm(total=total_samples) as pbar:
        for count, future in futures:
            batch_embeddings = ray.get(future)
            embeddings.extend(batch_embeddings)
            pbar.update(count)
    
    # Convert to array and ensure correct order
    embeddings = np.array(embeddings)
    
    # Create output directory
    output_dir = config.embedding.output_path
    makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    embedding_file = os.path.join(output_dir, "embeddings.npy")
    print(f"Saving embeddings to {embedding_file}")
    print(f"Embedding shape: {embeddings.shape}")
    np.save(embedding_file, embeddings)
    
    print("Done!")

if __name__ == '__main__':
    main()
