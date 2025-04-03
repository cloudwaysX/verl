import random
import torch
from torch.utils.data import Sampler, BatchSampler

import torch
from torch.utils.data import Sampler

class ScoreOrderedSampler(Sampler):
    def __init__(self, 
                 dataset_size,
                 selection_fn,
                 base_sampler,
                 score_threshold=None,
                 greedy_exploration_ratio=0.0,
                 descending=True):
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
        self.descending = descending
        self.base_sampler = base_sampler
        self.greedy_exploration_ratio = greedy_exploration_ratio
        self._iter_count = 0
        self.seed = 42
        
        print(f"ScoreOrderedSampler: score_threshold={self.score_threshold}, "
              f"greedy_exploration_ratio={self.greedy_exploration_ratio}, "
              f"descending={self.descending}")

    def __iter__(self):
        if self._iter_count == 0:
            print("First iteration: using provided base sampler")
            # For the first iteration, use the provided base sampler
            for idx in self.base_sampler:
                yield idx
        else:
            # iteration_str = "First iteration" if self._iter_count == 0 else f"Iteration {self._iter_count + 1}"
            # print(f"{iteration_str}: score-ordered sampling")
            
            # Calculate and sort scores - we recalculate each iteration
            all_indices = list(range(self.dataset_size))
            indices_with_scores = [(idx, self.selection_fn(idx)) for idx in all_indices]
            sorted_indices_with_scores = sorted(
                indices_with_scores,
                key=lambda x: x[1],  # Sort by score
                reverse=self.descending
            )
            
            # Separate indices and scores
            sorted_indices = [idx for idx, _ in sorted_indices_with_scores]
            sorted_scores = [score for _, score in sorted_indices_with_scores]
            
            random.seed(self.seed + self._iter_count)
            
            # Yield indices in sorted order until threshold is reached
            for i, idx in enumerate(sorted_indices):
                # Check if we've crossed the score threshold
                if self.score_threshold is not None:
                    score = sorted_scores[i]
                    if (self.descending and score <= self.score_threshold) or \
                       (not self.descending and score >= self.score_threshold):
                        if self.greedy_exploration_ratio == 0.0:
                            print(f"As the {i} the sample, score threshold reached: {score} < {self.score_threshold}")
                            break
                        elif random.random() < self.greedy_exploration_ratio:
                            continue
                
                yield idx
        
        self._iter_count += 1
        
    def __len__(self):
        return self.dataset_size
    
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
        self._iter_count = 0
        

    def __iter__(self):
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
        self._iter_count += 1
        
    def __len__(self):
        return len(self.base_batch_sampler)
    
    def save_state(self):
        return {'iter_count': self._iter_count}
    
    def load_state(self, state):
        self._iter_count = state['iter_count']
