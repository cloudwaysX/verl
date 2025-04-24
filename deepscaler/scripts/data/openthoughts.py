"""Script to prepare OpenR1-Math-220k dataset for training.

This script processes the OpenR1-Math-220k dataset into a standardized format for
training and testing math reasoning models. It loads problems from the dataset,
adds instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs

from datasets import load_dataset


def make_map_fn(split: str, source: str = None):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')
        source: Source dataset name

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        # Extract the problem statement
        problem = example.get('problem', '')
        
        # Add a step-by-step instruction
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        
        # Combine problem and instruction
        question = f"{problem} {instruction}"
        
        # Get the answer from the 'answer' field
        answer = example.get('answer', '')
        
        # Get problem_type for ability field
        problem_type = example.get('problem_type', 'math')
        
        # Calculate difficulty as 6 minus correctness_count
        correctness_count = example.get('correctness_count', 0)
        difficulty = 6 - correctness_count
        
        # Construct the data in the format expected by the training pipeline
        data = {
            "data_source": source or "OpenR1-Math-220k",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": problem_type,
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'difficulty': difficulty,
            }
        }
        return data
    
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process OpenR1-Math-220k dataset for training')
    parser.add_argument(
        '--local_dir',
        default=os.path.join(os.environ.get('HOME', ''), 'data'),
        help="Path to save the processed data files"
    )
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.dataset_path}...")
    
from datasets import load_dataset

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("bethgelab/CuratedThoughts", "OpenR1-Math-220k-default", split="train")
    
    print(f"Train dataset size: {len(train_dataset)}")
    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train', args.dataset_path)
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    # Save training dataset
    print(f"Processed train data size: {len(train_data)}")
    if len(train_data) > 0:
        train_df = pd.DataFrame(train_data)
        makedirs(os.path.join(local_dir, "openr1-math"), exist_ok=True)
        train_path = os.path.join(local_dir, 'openr1-math/train.parquet')
        train_df.to_parquet(train_path)
        print(f"Saved train data to {train_path}")

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        print(f"Copying data to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir, exist_ok=True)
        copy(src=os.path.join(local_dir, "openr1-math"), dst=os.path.join(hdfs_dir, "openr1-math"))
        print("HDFS copy complete")
    
    print("Processing complete!")