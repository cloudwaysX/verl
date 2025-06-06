import random
import torch
import numpy as np
from torch.utils.data import Sampler, BatchSampler

import torch
from torch.utils.data import Sampler

class ScoreOrderedSampler(Sampler):
    def __init__(self, 
                 dataset_size,
                 selection_fn,
                 base_sampler,
                 score_threshold=None,
                 size_threshold=None,
                 greedy_exploration_ratio=0.0,
                 descending=True,
                 resume=False,
                 shuffled=False,
                 dynamic_threshold=False,
                 dynamaic_threshold_params=None,
                 min_iter_size=0):
        """
        A sampler that yields indices ordered by score after the first iteration.
        First iteration uses the provided base_sampler if available.
        
        Args:
            dataset_size (int): Size of the dataset.
            selection_fn (callable): A function that takes an index and returns a numeric selection metric.
            score_threshold (float, optional): If provided, stop iteration when scores fall below this threshold.
            descending (bool): If True, sort in descending order (highest to lowest scores).
            base_sampler (Sampler, optional): Sampler to use for first iteration. If None, uses range(dataset_size).
        """
        self.dataset_size = dataset_size
        self.selection_fn = selection_fn
        self.score_threshold = score_threshold
        self.size_threshold = size_threshold
        self.descending = descending
        self.base_sampler = base_sampler
        self.greedy_exploration_ratio = greedy_exploration_ratio
        self._iter_count = -1
        self.seed = 42
        self.shuffled = shuffled
        self.dynamic_threshold = dynamic_threshold
        self.dynamaic_threshold_params = dynamaic_threshold_params
        self.resume = resume
        self.min_iter_size = min_iter_size
        
        print(f"ScoreOrderedSampler: score_threshold={self.score_threshold}, "
              f"greedy_exploration_ratio={self.greedy_exploration_ratio}, "
              f"descending={self.descending}")
        
    def _compute_threhold_dynamics(self):
        avg_score = 0.0
        for idx in range(self.dataset_size):
            avg_score += self.selection_fn(idx)
        avg_score /= self.dataset_size
        score_gap = self.score_threshold[1] - self.score_threshold[0] #[lower_bound, upper_bound]
        alpha = self.dynamaic_threshold_params.get("alpha",5.0)
        eta = self.score_threshold[0]
        # The lower bound starts from the original lower bound and decreases with the average score increasing
        new_lower_bound = np.max(self.score_threshold[0] - eta * np.tanh(alpha * avg_score), 0.0)
        new_upper_bound = np.min(new_lower_bound + score_gap, 1.0)
        self.score_threshold = (new_upper_bound, new_lower_bound)
        
    def _calculate_included_indices(self):
        """Calculate which indices will be included in the current iteration."""
        # Set random seed for exploration decisions
        random.seed(self.seed + self._iter_count)
        
        # Calculate scores for all indices
        all_indices = list(range(self.dataset_size))
        indices_with_scores = [(idx, self.selection_fn(idx)) for idx in all_indices]
        
        # Sort indices based on scores
        sorted_indices_with_scores = sorted(
            indices_with_scores,
            key=lambda x: x[1],  # Sort by score
            reverse=self.descending
        )
        
        # Extract just the indices
        sorted_indices = [idx for idx, _ in sorted_indices_with_scores]
        sorted_scores = [score for _, score in sorted_indices_with_scores]
        
        if self.score_threshold is None:
            # No threshold, include all indices
            if self.size_threshold is not None:
                tmp = min(self.size_threshold,
                          int(self.size_threshold*len(sorted_indices)))
                sorted_indices = sorted_indices[:tmp]
            return sorted_indices

        # Find the split point (first index below threshold)
        # And indices included in [split_idx[0], split_idx[1]) are being selected
        split_idx = [0, len(sorted_scores)]
        upper_bound = self.score_threshold[1]
        lower_bound = self.score_threshold[0]
        print("DEBUG",split_idx)
        for i, score in enumerate(sorted_scores):
            if self.descending:
                if score > upper_bound:
                    split_idx[0] = i+1
                elif score <= lower_bound:
                    split_idx[1] = i
                    print("DEBUG lower bound", lower_bound, self._iter_count)
                    print("DEBUG2",split_idx)
                    break
            else:
                raise NotImplementedError("Ascending order not implemented yet.")
        else:
            # No indices below threshold
            return sorted_indices
        
        # Split into above and below threshold
        within_threshold = sorted_indices[split_idx[0]:split_idx[1]]
        outside_threshold = sorted_indices[split_idx[1]:] + sorted_indices[:split_idx[0]]
        
        # Randomly select a subset of below-threshold samples
        explore_count = int(len(outside_threshold) * self.greedy_exploration_ratio)
        exploration_samples = random.sample(outside_threshold, explore_count) if explore_count > 0 else []
        
        # Combine above-threshold and exploration samples
        print(f"{len(within_threshold)} samples above threshold.")
        included_indices = within_threshold 
        
        # Re-sort the included indices by score
        idx_to_score = {idx: score for idx, score in zip(sorted_indices, sorted_scores)}
        included_indices.sort(key=lambda idx: idx_to_score[idx], reverse=self.descending)
        # Here we do not sort the exploration samples, so they will be in random order
        included_indices = included_indices + exploration_samples
        if len(included_indices) < self.min_iter_size:
            print(f"Not enough samples, adding {self.min_iter_size - len(included_indices)} random samples")
            included_indices = included_indices + random.sample(outside_threshold, self.min_iter_size - len(included_indices))
        
        return included_indices

    def __iter__(self):
        self._iter_count += 1
        if self.dynamic_threshold:
            self._compute_threhold_dynamics()
        if self._iter_count == 0 and not self.resume:
            print("First iteration: using provided base sampler")
            # For the first iteration, use the provided base sampler
            for idx in self.base_sampler:
                yield idx
        else:
            included_indices = self._calculate_included_indices()
            self._current_included_indices = included_indices
            print(f"Current iteration {self._iter_count}: {len(included_indices)} included indices")
            if self.shuffled:
                random.seed(self.seed + self._iter_count+1234)
                random.shuffle(included_indices)
            for idx in included_indices:
                yield idx
        
    def __len__(self):
        if self._iter_count <= 0:
            return len(self.base_sampler)
        else:
            return len(self._current_included_indices)
    
    def save_state(self):
        return {'iter_count': self._iter_count}
    
    def load_state(self, state):
        self._iter_count = state['iter_count']

class GreedyBatchSampler(Sampler):
    def __init__(self, 
                 base_batch_sampler, 
                 selection_fn, 
                 greedy_top_percent, 
                 greedy_exploration_ratio):
        """
        Args:
            base_batch_sampler (BatchSampler): Yields batches of indices (expected size = base_batch_size*2).
            selection_fn (callable): A function that takes an index and returns a numeric selection metric.
            greedy_top_percent (float): A float between 0 and 0.5. For example, 0.0 means always pick from the top 50%.
            greedy_exploration_ratio (float): With this probability, randomly select half the indices.
        """
        self.base_batch_sampler = base_batch_sampler
        self.selection_fn = selection_fn
        self.greedy_top_percent = greedy_top_percent
        self.greedy_exploration_ratio = greedy_exploration_ratio
        self._iter_count = -1
        

    def __iter__(self):
        self._iter_count += 1
        for batch in self.base_batch_sampler:
            half = len(batch) // 2  # we want to keep half of the indices
            if self._iter_count<=1:
                print("Initial epochs, select 50%")
                sorted_batch = sorted(batch, key=lambda idx: self.selection_fn(idx), reverse=True)
                # Skip a fraction at the beginning defined by greedy_top_percent.
                selected = sorted_batch[:half]
            elif random.random() < self.greedy_exploration_ratio:
                # Randomly sample half the batch indices.
                print(f"With prob {self.greedy_exploration_ratio}, randomly select half the indices.")
                selected = random.sample(batch, half)
            else:
                print(f"With prob {1-self.greedy_exploration_ratio}, select top {self.greedy_top_percent*100}% to {self.greedy_top_percent*100+50}%")   
                # Sort the batch indices by the selection metric in descending order.
                sorted_batch = sorted(batch, key=lambda idx: self.selection_fn(idx), reverse=True)
                # Skip a fraction at the beginning defined by greedy_top_percent.
                start = int(self.greedy_top_percent * len(batch))
                selected = sorted_batch[start:start+half]
            yield selected
        
    def __len__(self):
        return len(self.base_batch_sampler)
    
    def save_state(self):
        return {'iter_count': self._iter_count}
    
    def load_state(self, state):
        self._iter_count = state['iter_count']
