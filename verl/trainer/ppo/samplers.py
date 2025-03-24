import random
import torch
from torch.utils.data import Sampler, BatchSampler

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
        self.is_initial_epoch = True
        
    def set_inital_epoch(self, initial_epoch):
        self.is_initial_epoch = initial_epoch

    def __iter__(self):
        for batch in self.base_batch_sampler:
            half = len(batch) // 2  # we want to keep half of the indices
            if self.is_initial_epoch:
                print("Initial epochs, select 50%")
                sorted_batch = sorted(batch, key=lambda idx: self.selection_fn(idx), reverse=True)
                # Skip a fraction at the beginning defined by greedy_top_percent.
                start = int(self.greedy_top_percent * len(batch))
                selected = sorted_batch[:half]
            elif random.random() < self.greedy_exploration_ratio:
                # Randomly sample half the batch indices.
                selected = random.sample(batch, half)
            else:
                print(f"With prob {1-p}, select top {self.greedy_top_percent*100}% to {self.greedy_top_percent*100+50}%")   
                # Sort the batch indices by the selection metric in descending order.
                sorted_batch = sorted(batch, key=lambda idx: self.selection_fn(idx), reverse=True)
                # Skip a fraction at the beginning defined by greedy_top_percent.
                start = int(self.greedy_top_percent * len(batch))
                selected = sorted_batch[start:start+half]
            yield selected
    def __len__(self):
        return len(self.base_batch_sampler)
