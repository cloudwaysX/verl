import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Analyze correlation between embedding distances and variance differences')
    parser.add_argument('--embedding_path', type=str, required=True, help='Path to embeddings.npy file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset parquet file')
    parser.add_argument('--variance_path', type=str, required=True, help='Path to variance.json file')
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
    
    print(f"Dataset size after sampling: {len(dataset)}")
    print(f"Embeddings shape after sampling: {embeddings.shape}")
    
    # Step 3: Load variance data
    print(f"Loading variance data from {args.variance_path}")
    with open(args.variance_path, 'r') as f:
        variance_data = json.load(f)
    variances = np.array(variance_data)
        
    assert len(variances) == len(embeddings), "Variance data doesn't match embeddings size"
    
    print(f"Loaded {len(variances)} variance values")
    
    # Step 4: Sample pairwise distances and variance differences
    print(f"Sampling {args.n_pairs} pairs for analysis")
    np.random.seed(42)  # For reproducibility
    idx = np.random.choice(len(embeddings), size=(args.n_pairs, 2), replace=True)
    
    # Calculate distances and variance differences
    dists = cosine_distances(embeddings[idx[:, 0]], embeddings[idx[:, 1]]).diagonal()
    delta_v = np.abs(variances[idx[:, 0]] - variances[idx[:, 1]])
    
    # Step 5: Compute correlation
    corr, pval = spearmanr(dists, delta_v)
    print(f"Spearman correlation: ρ={corr:.3f}, p={pval:.3g}")
    
    # Save correlation results
    with open(os.path.join(args.output_dir, 'correlation_results.json'), 'w') as f:
        json.dump({
            'spearman_rho': float(corr),
            'p_value': float(pval),
            'n_pairs': args.n_pairs
        }, f, indent=2)
    
    # Step 6: Create visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(dists, delta_v, alpha=0.1)
    plt.xlabel("Embedding Distance (Cosine)")
    plt.ylabel("|Variance Difference|")
    plt.title(f"Relationship between Embedding Distance and Variance Difference\nSpearman ρ={corr:.3f}, p={pval:.3g}")
    
    # Add trend line
    z = np.polyfit(dists, delta_v, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(dists), p(np.sort(dists)), "r--", alpha=0.8)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'distance_vs_variance.png'), dpi=300)
    plt.close()
    
    # Additional analysis: Check correlation at different distance ranges
    print("Analyzing correlation at different distance ranges...")
    
    n_bins = 5
    dist_bins = np.linspace(np.min(dists), np.max(dists), n_bins+1)
    
    bin_results = []
    for i in range(n_bins):
        bin_mask = (dists >= dist_bins[i]) & (dists < dist_bins[i+1])
        if np.sum(bin_mask) > 10:  # Only calculate if we have enough samples
            bin_dists = dists[bin_mask]
            bin_deltas = delta_v[bin_mask]
            bin_corr, bin_pval = spearmanr(bin_dists, bin_deltas)
            bin_results.append({
                'distance_range': f"{dist_bins[i]:.2f}-{dist_bins[i+1]:.2f}",
                'n_samples': int(np.sum(bin_mask)),
                'spearman_rho': float(bin_corr),
                'p_value': float(bin_pval)
            })
            print(f"Distance range {dist_bins[i]:.2f}-{dist_bins[i+1]:.2f}: ρ={bin_corr:.3f}, p={bin_pval:.3g}, n={np.sum(bin_mask)}")
    
    # Save bin results
    with open(os.path.join(args.output_dir, 'distance_bin_analysis.json'), 'w') as f:
        json.dump(bin_results, f, indent=2)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()