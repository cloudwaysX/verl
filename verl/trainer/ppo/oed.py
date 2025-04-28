import numpy as np
from sklearn.metrics import pairwise_distances
import os
from tqdm import tqdm
import torch


# Coreset selection
def coreset_selection(embeddings: np.ndarray, 
                      size: int, 
                      oed_save_path: str = None, 
                      random_seed: int = None,
                      mode="cpu") -> list[int]:
    """
    Select `size` points from `embeddings` via farthest‐first (coreset) traversal.

    Args:
        embeddings: array of shape (n_samples, embedding_dim)
        size: number of points to select

    Returns:
        selected_idxs: list of length `size` with the indices of the selected embeddings
    """
    
    os.makedirs(os.path.dirname(oed_save_path), exist_ok=True)
    if random_seed is None:
        cache_file = os.path.join(os.path.dirname(oed_save_path), 'orderd_coreset_idxs.npy')
    else:
        cache_file = os.path.join(os.path.dirname(oed_save_path), f'orderd_coreset_idxs_{random_seed}.npy')
    if os.path.exists(cache_file):
        print(f"Loading coreset selection from {cache_file}")
        # Because the order is deterministic, we can just compute the selection once and save it.
        ordered_idxs = np.load(cache_file)
        selected_idxs = ordered_idxs[:size]
        print(f"The first 100 selected ids are: {selected_idxs[:100]}")
        return selected_idxs
      
    if mode == "cpu":
        ordered_idxs = corset_selection_cpu(embeddings, len(embeddings)//2, random_seed)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    np.save(cache_file, ordered_idxs)
    selected_idxs = ordered_idxs[:size]
    print(f"The first 100 selected ids are: {selected_idxs[:100]}")
    return selected_idxs

def corset_selection_cpu(embeddings: np.ndarray, size: int, random_seed: int = None) -> list[int]:
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
        if i == 0 and random_seed is not None:
            np.random.seed(random_seed)
            next_idx = np.random.choice(n_samples, 1, replace=False)[0]
        else:
            next_idx = int(np.argmax(min_dist)) # Will just choose the firt one
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
  
  

# Reverse coreset selection
def reverse_coreset_selection(embeddings: np.ndarray, 
                              size: int, 
                              oed_save_path: str = None, 
                              random_seed: int = None, 
                              mode="cpu") -> list[int]:

    os.makedirs(os.path.dirname(oed_save_path), exist_ok=True)
    if random_seed is None:
        cache_file = os.path.join(os.path.dirname(oed_save_path), 'orderd_reversed_coreset_idxs.npy')
    else:
        cache_file = os.path.join(os.path.dirname(oed_save_path), f'orderd_reversed_coreset_idxs_{random_seed}.npy')
    if os.path.exists(cache_file):
        print(f"Loading coreset selection from {cache_file}")
        # Because the order is deterministic, we can just compute the selection once and save it.
        ordered_idxs = np.load(cache_file)
        selected_idxs = ordered_idxs[:size]
        print(f"The first 100 selected ids are: {selected_idxs[:100]}")
        return selected_idxs
      
    if mode == "cpu":
        ordered_idxs = reverse_coreset_selection_cpu(embeddings, len(embeddings)//2, random_seed)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    np.save(cache_file, ordered_idxs)
    selected_idxs = ordered_idxs[:size]
    print(f"The first 100 selected ids are: {selected_idxs[:100]}")
    return selected_idxs
  
def reverse_coreset_selection_cpu(embeddings: np.ndarray, size: int, random_seed: int = None) -> list[int]:
    """
    Select `size` points from `embeddings` via farthest‐first (coreset) traversal.
    """
    
    X = embeddings
    n_samples = X.shape[0]
    
    # … before the loop …
    min_dist = np.full(n_samples, np.inf, dtype=float)
    selected_idxs: list[int] = []
    taken = np.zeros(n_samples, dtype=bool)
    
    bar = tqdm(total=size,
               desc="Selecting reversed coreset",
               unit="pt",
               leave=False)

    for i in range(size):
        if i == 0 and random_seed is not None:
            np.random.seed(random_seed)
            next_idx = np.random.choice(n_samples, 1, replace=False)[0]
        elif i == 0:
            # deterministic seed if you like
            next_idx = 0
        else:
            # mask out already picked points
            masked_dist = np.where(taken, np.inf, min_dist)
            # pick the point _closest_ to the existing set
            next_idx = int(np.argmin(masked_dist))

        selected_idxs.append(next_idx)
        taken[next_idx] = True

        # compute distances to the freshly picked point
        dists = pairwise_distances(X, X[next_idx].reshape(1, -1)).reshape(-1)
        # update the distance‐to‐nearest‐center
        min_dist = np.minimum(min_dist, dists)

        if i % 100 == 0:
          bar.update(100)
    bar.close()
    return selected_idxs



