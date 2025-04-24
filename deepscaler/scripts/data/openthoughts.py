"""Script to prepare OpenR1-Math-220k dataset for training.

This script processes the OpenR1-Math-220k dataset into a standardized format for
training and testing math reasoning models. It loads problems from the dataset,
adds instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any
from collections import Counter

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs

from datasets import load_dataset


def make_map_fn(split: str,):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')
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
            "data_source": "OpenR1-Math-220k",
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


def print_dataset_info(dataset, name="Dataset"):
    """Print information about the dataset."""
    print(f"\n{name} Information:")
    print(f"Number of examples: {len(dataset)}")
    
    # Get all field names
    if len(dataset) > 0:
        example = dataset[0]
        print(f"Fields: {list(example.keys())}")
    
    # Print dataset statistics
    if 'problem_type' in example:
        problem_types = Counter([item.get('problem_type', 'unknown') for item in dataset])
        print("\nProblem Type Distribution:")
        for problem_type, count in problem_types.most_common():
            print(f"  {problem_type}: {count} ({count/len(dataset)*100:.2f}%)")
    
    if 'correctness_count' in example:
        correctness_counts = Counter([item.get('correctness_count', 0) for item in dataset])
        print("\nCorrectness Count Distribution:")
        for count, freq in sorted(correctness_counts.items()):
            print(f"  {count}: {freq} examples ({freq/len(dataset)*100:.2f}%)")
            
    # Calculate difficulty distribution
    if 'difficulty' in example['extra_info']:
        difficulties = Counter([item['extra_info']['difficulty'] for item in dataset])
        print("\nDifficulty Distribution:")
        for difficulty, count in sorted(difficulties.items()):
            print(f"  {difficulty}: {count} ({count/len(dataset)*100:.2f}%)")

def print_examples(dataset, num_examples=3):
    """Print example problems from the processed dataset."""
    print(f"\nExample Problems (showing {min(num_examples, len(dataset))} examples):")
    
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"\nExample #{i+1}:")
        
        # Print prompt
        prompt = example["prompt"][0]["content"]
        # Truncate if too long
        if len(prompt) > 300:
            prompt = prompt[:297] + "..."
        print(f"Prompt: {prompt}")
        
        # Print ability and ground truth answer
        print(f"Ability: {example['ability']}")
        print(f"Ground Truth: {example['reward_model']['ground_truth']}")
        
        # Print difficulty
        print(f"Difficulty: {example['extra_info']['difficulty']}")
        
        print("-" * 80)

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

    print(f"Loading dataset from bethgelab/CuratedThoughts...")

    # Login using e.g. `huggingface-cli login` to access this dataset
    train_dataset = load_dataset("bethgelab/CuratedThoughts", "OpenR1-Math-220k-default", split="train")
    
    print(f"Train dataset size: {len(train_dataset)}")
    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    # Print train dataset info and examples
    print_dataset_info(train_data, "Processed Train Data")
    print_examples(train_data, 3)

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
