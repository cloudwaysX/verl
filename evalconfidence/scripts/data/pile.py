import os
import argparse
import datasets

# Uncomment the following lines if you have HDFS helper functions.
# from verl.utils.hdfs_io import copy, makedirs

def process_pile_example(example, split, idx):
    """
    Transform an example from the Pile dataset into an SFT-style dictionary
    with the following structure:
      - data_source: always "pile"
      - ability: "general"
      - text: the full text from the example
      - extra_info: contains only the split and index.
    """
    new_example = {
        "data_source": "pile",
        "ability": "general",
        "text": example.get("text", ""),
        "extra_info": {
            "split": split,
            "index": idx
        }
    }
    return new_example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/pile', help='Local directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory to copy processed data to (optional)')
    parser.add_argument('--split', default='train', help='Dataset split to process (e.g., train, validation, test)')
    args = parser.parse_args()
    
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Load the Pile dataset from Hugging Face. Adjust the dataset name/config if needed.
    dataset = datasets.load_dataset("monology/pile-uncopyrighted", split=args.split, trust_remote_code=True)

    chunk_size = 50_000 # Adjust this number based on your memory constraints.
    local_path = os.path.join(local_dir, f"{args.split}.jsonl")

    with open(local_path, 'w') as out_file:
        # Assuming dataset is a list-like object; for streaming datasets, adjust accordingly.
        for start_idx in range(0, len(dataset), chunk_size):
            chunk = dataset[start_idx: start_idx + chunk_size]
            for idx, example in enumerate(chunk, start=start_idx):
                processed = process_pile_example(example, args.split, idx)
                out_file.write(json.dumps(processed) + "\n")
            print(f"Processed and saved chunk starting at index {start_idx}")
    
    # Optionally copy the output to HDFS.
    if args.hdfs_dir is not None:
        # Uncomment the following lines if you have HDFS helper functions.
        # makedirs(args.hdfs_dir)
        # copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied processed data to HDFS directory: {args.hdfs_dir}")

if __name__ == '__main__':
    main()
