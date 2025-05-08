import os
import json
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Analyze correlation between embedding distances and acc differences')
    parser.add_argument('--embedding_path', type=str, required=True, help='Path to embeddings.npy file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset parquet file')
    parser.add_argument('--acc_path', type=str, required=True, help='Path to acc.json file')
    parser.add_argument('--output_dir', type=str, default='./analysis_results', help='Directory to save analysis results')
    parser.add_argument('--train_ratio', type=float, default=1.0, help='Ratio of dataset to use, matching main_generation')
    parser.add_argument('--train_ratio_seed', type=int, default=None, help='Random seed for sampling, should match main_generation')
    parser.add_argument('--n_pairs', type=int, default=10000, help='Number of pairs to sample for analysis')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load embeddings
    print(f"Loading embeddings from {args.embedding_path}")
    embeddings = np.load(args.embedding_path)
    n_size, n_dim = embeddings.shape
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # Step 2: Load and sample dataset matching the same sampling as in main_generation
    print(f"Loading dataset from {args.data_path}")
    dataset = pd.read_parquet(args.data_path)
    original_len = len(dataset)
    print(f"Original dataset size: {original_len}")

    assert original_len == n_size, "Dataset size doesn't match embeddings size"
    # Apply the same sampling as in main_generation
    if args.train_ratio < 1:
        size = int(original_len * args.train_ratio)
        print("Assuming embeddings and dataset are aligned and sampling embeddings accordingly...")
        if args.train_ratio_seed is not None:
            np.random.seed(args.train_ratio_seed)
            indices = np.random.permutation(original_len)[:size]
        else:
            indices = np.arange(size)
        embeddings = embeddings[indices]

    print(f"Dataset size after sampling (implied by embeddings): {embeddings.shape[0]}")
    print(f"Embeddings shape after sampling: {embeddings.shape}")

    # Step 3: Load acc data
    print(f"Loading acc data from {args.acc_path}")
    with open(args.acc_path, 'r') as f:
        acc_data = json.load(f)
    accs = np.array(acc_data)

    assert len(accs) == len(embeddings), "Acc data doesn't match embeddings size after sampling"

    print(f"Loaded {len(accs)} acc values, matching sampled embeddings size")

    # Step 4: Sample pairwise distances and acc differences
    print(f"Sampling {args.n_pairs} pairs for analysis")
    np.random.seed(42)  # For reproducibility
    idx = np.random.choice(len(embeddings), size=(args.n_pairs, 2), replace=True)

    # Calculate distances and acc differences
    dists = cosine_distances(embeddings[idx[:, 0]], embeddings[idx[:, 1]]).diagonal()
    delta_a = np.abs(accs[idx[:, 0]] - accs[idx[:, 1]])

    # Step 5: Compute overall correlation
    corr, pval = spearmanr(dists, delta_a)
    print(f"Overall Spearman correlation (Distance vs. Acc Difference): ρ={corr:.3f}, p={pval:.3g}") # Modified print

    # Save overall correlation results
    with open(os.path.join(args.output_dir, 'overall_correlation_results.json'), 'w') as f:
        json.dump({
            'spearman_rho': float(corr),
            'p_value': float(pval),
            'n_pairs': args.n_pairs
        }, f, indent=2)
    """
    # Step 6: Create visualization # Keep commented
    plt.figure(figsize=(10, 8))
    plt.scatter(dists, delta_a, alpha=0.1)
    plt.xlabel("Embedding Distance (Cosine)")
    plt.ylabel("|Acc Difference|")
    plt.title(f"Relationship between Embedding Distance and Acc Difference\nSpearman ρ={corr:.3f}, p={pval:.3g}")

    # Add trend line
    z = np.polyfit(dists, delta_a, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(dists), p(np.sort(dists)), "r--", alpha=0.8)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'distance_vs_acc.png'), dpi=300)
    plt.close()
    """
    # Additional analysis 1: Check correlation at different distance ranges
    print("\nAnalyzing correlation at different embedding distance ranges...") # Added newline and clarified print

    n_bins = 3
    dist_bins = np.linspace(np.min(dists), np.max(dists), n_bins+1)

    distance_bin_results = [] # Changed variable name
    for i in range(n_bins):
        bin_mask = (dists >= dist_bins[i]) & (dists < dist_bins[i+1])
        if np.sum(bin_mask) > 10:  # Only calculate if we have enough samples
            bin_dists = dists[bin_mask]
            bin_deltas = delta_a[bin_mask]
            bin_corr, bin_pval = spearmanr(bin_dists, bin_deltas)
            distance_bin_results.append({ # Changed variable name
                'distance_range': f"{dist_bins[i]:.4f}-{dist_bins[i+1]:.4f}", # Added format
                'n_samples': int(np.sum(bin_mask)),
                'spearman_rho': float(bin_corr),
                'p_value': float(bin_pval)
            })
            print(f"Distance range {dist_bins[i]:.4f}-{dist_bins[i+1]:.4f}: ρ={bin_corr:.3f}, p={bin_pval:.3g}, n={np.sum(bin_mask)}") # Added format

    # Save distance bin results
    with open(os.path.join(args.output_dir, 'distance_bin_analysis.json'), 'w') as f: # Changed filename
        json.dump(distance_bin_results, f, indent=2) # Changed variable name

    # Additional analysis 2: Check correlation at different acc difference ranges
    print("\nAnalyzing correlation at different acc difference ranges...") # Added newline and clarified print

    n_bins = 2 # n_bins can be reused or set separately if needed
    acc_diff_bins = np.linspace(np.min(delta_a), np.max(delta_a), n_bins+1)

    acc_diff_bin_results = [] # Changed variable name
    for i in range(n_bins):
        # Create mask based on acc difference bins
        bin_mask = (delta_a >= acc_diff_bins[i]) & (delta_a < acc_diff_bins[i+1])
        if np.sum(bin_mask) > 10:  # Only calculate if we have enough samples
            # Extract distances and acc differences for this bin
            bin_dists = dists[bin_mask]
            bin_deltas = delta_a[bin_mask]
            # Compute correlation within this acc difference bin
            bin_corr, bin_pval = spearmanr(bin_dists, bin_deltas)
            acc_diff_bin_results.append({ # Changed variable name
                'acc_difference_range': f"{acc_diff_bins[i]:.4f}-{acc_diff_bins[i+1]:.4f}",
                'n_samples': int(np.sum(bin_mask)),
                'spearman_rho': float(bin_corr),
                'p_value': float(bin_pval)
            })
            print(f"Acc Diff range {acc_diff_bins[i]:.4f}-{acc_diff_bins[i+1]:.4f}: ρ={bin_corr:.3f}, p={bin_pval:.3g}, n={np.sum(bin_mask)}")

    # Save acc difference bin results
    with open(os.path.join(args.output_dir, 'acc_diff_bin_analysis.json'), 'w') as f: # Changed filename
        json.dump(acc_diff_bin_results, f, indent=2) # Changed variable name

    print(f"\nAnalysis complete. Results saved to {args.output_dir}") # Added newline

if __name__ == "__main__":
    main()
