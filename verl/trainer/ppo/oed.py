import numpy as np
from sklearn.metrics import pairwise_distances
import os
from tqdm import tqdm
import torch

def coreset_selection(embeddings: np.ndarray, size: int, oed_save_path: str = None, mode="cpu") -> list[int]:
    """
    Select `size` points from `embeddings` via farthest‐first (coreset) traversal.

    Args:
        embeddings: array of shape (n_samples, embedding_dim)
        size: number of points to select

    Returns:
        selected_idxs: list of length `size` with the indices of the selected embeddings
    """
    
    os.makedirs(os.path.dirname(oed_save_path), exist_ok=True)
    cache_file = os.path.join(os.path.dirname(oed_save_path), 'coreset_idxs.npy')
    if os.path.exists(cache_file):
        print(f"Loading coreset selection from {cache_file}")
        selected_idxs = np.load(cache_file)
        return selected_idxs
      
    if mode == "cpu":
      selected_idxs = corset_selection_cpu(embeddings, size)
    elif mode == "gpu":
        selected_idxs = coreset_selection_gpu(embeddings, size)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    np.save(cache_file, selected_idxs)
    return selected_idxs

def corset_selection_cpu(embeddings: np.ndarray, size: int) -> list[int]:
    """
    Select `size` points from `embeddings` via farthest‐first (coreset) traversal.
    """
    
    X = embeddings
    n_samples = X.shape[0]

    # 1. Initialize an array of “closest‐center distances”:
    #    For each point i, min_dist[i] will track the distance
    #    to the nearest center chosen so far. We start with +∞.
    min_dist = np.full(n_samples, np.inf, dtype=float)

    # 2. This list will hold the indices of the centers we pick.
    selected_idxs: list[int] = []

    # 3. Repeat until we’ve picked `size` centers:
    # create one bar; it will only redraw at most every `mininterval` seconds
    bar = tqdm(total=size,
               desc="Selecting coreset",
               unit="pt",
               leave=False)

    for i in range(size):
        # 3a. Find the point that is currently farthest from all chosen centers:
        #     that’s the argmax over min_dist.
        next_idx = int(np.argmax(min_dist))
        selected_idxs.append(next_idx)

        # 3b. Compute its distance to every point in X:
        #     pairwise_distances returns shape (n_samples, 1), reshape to (n_samples,)
        dists = pairwise_distances(X, X[next_idx].reshape(1, -1)).reshape(-1)

        # 3c. Update min_dist so that each point’s entry is now
        #     the minimum of its old value and its distance to this new center.
        #     In effect, min_dist[i] always equals:
        #       min ( over centers c chosen so far ) ‖X[i] – X[c]‖₂
        min_dist = np.minimum(min_dist, dists)
        
        if i%100 == 0:
            bar.update(100)

    bar.close()
    # 4. Return the list of selected indices.
    return selected_idxs

def coreset_selection_gpu(
    embeddings: np.ndarray,
    size: int,
    device: str = "cuda"
) -> list[int]:
    """
    GPU‐accelerated farthest‐first with proper cleanup.
    """
    # 1) Move to GPU in float32; wrap in no_grad so no graph is built
    with torch.no_grad():
        X = torch.from_numpy(embeddings).float().to(device)  # (n, d)
        n = X.size(0)
        min_dist = torch.full((n,), float("inf"), device=device)

        selected = []
        bar = tqdm(total=size,
                  desc="Selecting coreset",
                  unit="pt",
                  leave=False)
        for i in range(size):
            # pick farthest point
            idx = int(min_dist.argmax().item())
            selected.append(idx)

            # update distances in one big cdist
            d = torch.cdist(X, X[idx:idx+1], p=2).squeeze(1)  # (n,)
            min_dist = torch.minimum(min_dist, d)
            if i%100 == 0:
                bar.update(100)


    # 2) Cleanup: delete GPU tensors & sync + empty cache
    del X, min_dist
    # make sure all CUDA kernels are done
    torch.cuda.synchronize(device)
    # free as much as possible back to the allocator
    torch.cuda.empty_cache()

    return selected


