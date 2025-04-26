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
class BatchProcessor:
    def __init__(
        self,
        model_name,
        framework,
        max_length,
        pooling_method,
        normalize,
        prompt_name=None
    ):
        # Pin this actor to its assigned GPU
        gpu_ids = ray.get_gpu_ids()
        assert len(gpu_ids) == 1, f"Expected one GPU per actor, got {gpu_ids}"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        self.device = torch.device("cuda:0")

        self.framework = framework
        self.max_length = max_length
        self.pooling = pooling_method
        self.normalize = normalize
        self.prompt_name = prompt_name

        if framework == 'hf':
            # Load HF AutoModel once
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device).eval()
        elif framework == 'st':
            # Load SentenceTransformer once
            self.st_model = SentenceTransformer(model_name)
            if max_length is not None:
                self.st_model.max_seq_length = max_length
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def encode(self, batch_indices, batch_texts):
        """
        Encode a batch of texts and return (indices, embeddings)
        """
        if self.framework == 'hf':
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)

            if self.pooling == 'cls':
                embs = out.last_hidden_state[:, 0, :]
            else:
                tokens = out.last_hidden_state
                mask = encoded['attention_mask'].unsqueeze(-1).expand(tokens.size()).float()
                embs = torch.sum(tokens * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
            embs = embs.cpu().numpy()
            if self.normalize:
                embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        else:  # 'st'
            if self.prompt_name:
                embs = self.st_model.encode(batch_texts, prompt_name=self.prompt_name)
            else:
                embs = self.st_model.encode(batch_texts)
            if isinstance(embs, torch.Tensor):
                embs = embs.cpu().numpy()

        return batch_indices, embs

@hydra.main(config_path='config', config_name='inference', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Expose GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(config.embedding.n_gpus_per_node))
    ray.init() if not ray.is_initialized() else None

    # Setup
    framework = config.embedding.get('framework', 'hf')
    name = config.embedding.model.name
    model_path = copy_to_local(name) if framework == 'hf' else name

    # Read and sample
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
            elt = x[0]
            texts.append(elt['content'] if isinstance(elt, dict) and 'content' in elt else str(elt))
        else:
            texts.append(x)

    total = len(texts)
    indices = list(range(total))

    # Create actor pool (one per GPU)
    num_gpus = config.embedding.n_gpus_per_node * config.embedding.nnodes
    actors = [
        BatchProcessor.remote(
            model_path,
            framework,
            config.embedding.max_length,
            config.embedding.pooling_method,
            config.embedding.normalize_embeddings,
            config.embedding.get('prompt_name', None)
        )
        for _ in range(num_gpus)
    ]

    # Batch assignment
    bs = config.embedding.batch_size
    batches_idx = [indices[i:i+bs] for i in range(0, total, bs)]
    batches_txt = [[texts[i] for i in batch] for batch in batches_idx]

    # Dispatch to actors in round-robin
    futures = []
    for i, (idxs, txts) in enumerate(zip(batches_idx, batches_txt)):
        actor = actors[i % num_gpus]
        futures.append(actor.encode.remote(idxs, txts))

    # Collect
    results = ray.get(futures)
    dim = results[0][1].shape[1]
    embeddings = np.zeros((total, dim), dtype=results[0][1].dtype)
    for idxs, emb in results:
        embeddings[idxs, :] = emb

    # === Stats ===
    var_smp = np.var(embeddings, axis=1)
    print(f"Sample variance: mean={var_smp.mean():.4f}, std={var_smp.std():.4f}, min={var_smp.min():.4f}, max={var_smp.max():.4f}")
    cov = embeddings.T @ embeddings
    eigs = np.linalg.eigvalsh(cov)
    cond = eigs.max() / eigs.min()
    print(f"Eigenvalues: min={eigs.min():.4f}, max={eigs.max():.4f}, condition number={cond:.4f}")
    counts, bins = np.histogram(eigs, bins=50)
    print("Eigenvalue histogram:")
    for s,e,c in zip(bins[:-1], bins[1:], counts): print(f"[{s:.2e},{e:.2e}):{c}")

    # Save
    makedirs(config.embedding.output_path, exist_ok=True)
    out = os.path.join(config.embedding.output_path, 'embeddings.npy')
    np.save(out, embeddings)
    print(f"Saved embeddings to {out}, shape={embeddings.shape}")
