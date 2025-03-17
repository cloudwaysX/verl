import os
import argparse
import datasets

# Uncomment these if you need HDFS support.
# from verl.utils.hdfs_io import copy, makedirs

def process_math_qa_example(example, split, idx, data_source="allenai/math_qa"):
    """
    Transform an example from the math_qa dataset into an SFT-style dictionary,
    following the same principle as in gsm8k.py:
      - Remove original keys.
      - Build a new dictionary with the desired structure.
      - Use a short answer for reward_model["ground_truth"] and the full solution in extra_info["answer"].
    """
    # Build the prompt by combining the problem and its options.
    question_raw = f"{example['Problem']} Options: {example['options']}"
    question = question_raw  # You may add further instructions if desired.
    
    # Construct a new dictionary with only the desired keys.
    new_example = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["correct"]  # Only the short answer.
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": example["Rationale"].strip('"'),
            "question": example["Problem"],
        }
    }
    return new_example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math_qa', help='Local directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory to copy the processed data to (optional)')
    args = parser.parse_args()
    
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Load the math_qa dataset from Hugging Face.
    dataset = datasets.load_dataset("allenai/math_qa")
    
    # Process each split (train, test, validation) following the same principle as gsm8k.py.
    for split in dataset.keys():
        print(f"Processing split: {split} with {len(dataset[split])} examples.")
        
        processed_split = dataset[split].map(
            lambda example, idx: process_math_qa_example(example, split, idx),
            with_indices=True
        )
        
        local_path = os.path.join(local_dir, f"{split}.parquet")
        processed_split.to_parquet(local_path)
        print(f"Saved processed {split} split to {local_path}")
    
    # Optionally copy the output to HDFS.
    if args.hdfs_dir is not None:
        # Uncomment if you have HDFS helper functions.
        # makedirs(args.hdfs_dir)
        # copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied processed data to HDFS directory: {args.hdfs_dir}")

if __name__ == '__main__':
    main()
