import numpy as np
from sklearn.metrics import pairwise_distances
import os
from tqdm import tqdm

def coreset_selection(embeddings: np.ndarray, size: int, oed_save_path: str = None) -> list[int]:
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
    np.save(cache_file, np.array(selected_idxs, dtype=int))
    return selected_idxs

