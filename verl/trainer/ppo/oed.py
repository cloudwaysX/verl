import numpy as np
from sklearn.metrics import pairwise_distances
import os
from tqdm import tqdm
import torch

##################################################
# Coreset selection
##################################################
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
  
  
##################################################
# Reverse coreset selection
##################################################
def reverse_coreset_selection(embeddings: np.ndarray, 
                              size: int, 
                              oed_save_path: str = None, 
                              random_seed: int = None, 
                              initial_seedsamples_size: int = 1,
                              mode="cpu") -> list[int]:

    assert initial_seedsamples_size > 0, "initial_seedsamples_size must be greater than 0"
    os.makedirs(os.path.dirname(oed_save_path), exist_ok=True)
    if random_seed is None:
        cache_file = os.path.join(oed_save_path, f'orderd_reversed_coreset_initsize{initial_seedsamples_size}_idxs.npy')
    else:
        cache_file = os.path.join(oed_save_path, f'orderd_reversed_coreset_initsize{initial_seedsamples_size}_idxs_{random_seed}.npy')
    if os.path.exists(cache_file):
        print(f"Loading coreset selection from {cache_file}")
        # Because the order is deterministic, we can just compute the selection once and save it.
        ordered_idxs = np.load(cache_file)
        selected_idxs = ordered_idxs[:size]
        print(f"The first 100 selected ids are: {selected_idxs[:100]}")
        return selected_idxs
      
    if initial_seedsamples_size > 1:
        assert random_seed is not None, "random_seed must be specified when initial_seedsamples_size is greater than 1"
      
    if mode == "cpu":
        ordered_idxs = reverse_coreset_selection_cpu(
            embeddings, 
            len(embeddings)//2, 
            initial_seedsamples_size,
            random_seed)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    np.save(cache_file, ordered_idxs)
    selected_idxs = ordered_idxs[:size]
    print(f"The first 100 selected ids are: {selected_idxs[:100]}")
    return selected_idxs
  
def reverse_coreset_selection_cpu(
    embeddings: np.ndarray, 
    size: int, 
    initial_seedsamples_size: int = 1,
    random_seed: int = None) -> list[int]:
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
    
    if random_seed is not None:
        np.random.seed(random_seed)

    for i in range(size):
        # Find indices that haven't been taken yet
        available_indices = np.where(~taken)[0]

        if len(available_indices) == 0:
            print(
                "Warning: Ran out of unique samples during initial random "
                f"seeding at iteration {i}. Stopping early."
            )
            break # Stop if no available samples left
          
        if i < initial_seedsamples_size and random_seed is not None:
            # Choose one index randomly from the *available* ones
            # No need for replace=False since we select only 1 from available unique indices
            next_idx = np.random.choice(available_indices, 1)[0]

            # Note: The np.random.seed() call was removed from inside this block.
            # It's now set once before the loop for overall reproducibility.
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




###################################################
# RedAnt: Adaptive coverage sampling
###################################################
import json
def redant_selection(size: int, 
                     oed_save_path: str = None, 
                     random_seed: int = None, 
                     ) -> list[int]:
    """
    Select `size` points from `embeddings` via farthest‐first (coreset) traversal.
    """
 
    # RedAnt is not deterministic, so we cache the selection based on the size.
    cache_file = os.path.join(oed_save_path, f'redant_idxs_size{size}.json')

    if os.path.exists(cache_file):
        print(f"Loading coreset selection from {cache_file}")
        # Because the order is deterministic, we can just compute the selection once and save it.
        # Load indices from the JSON file
        with open(cache_file, 'r') as f:
            selected_idxs = json.load(f)
        # Ensure the loaded data is a list of integers
        if not isinstance(selected_idxs, list) or not all(isinstance(x, int) for x in selected_idxs):
             raise TypeError(f"Cached data in {cache_file} is not a list of integers.")
        return selected_idxs
    else:
        # The part that would generate the selection is removed as per the user's request
        # to only handle loading from cache.
        raise ValueError(
            f"RedAnt selection not cached for size {size} at {cache_file}. "
            "You need to run Google internal RedAnt first to generate the cache file."
            # Or, if running RedAnt is not possible, manually create the JSON file
            # at the specified path with a list of indices.
        )

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



