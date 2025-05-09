import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import argparse
import os

# Optional: Import UMAP if you have it installed
try:
    import umap
except ImportError:
    umap = None
    print("UMAP not installed. Install with 'pip install umap-learn' to use it.")


def load_embeddings_from_file(filepath):
    """
    Loads sample embeddings from a specified file.

    Args:
        filepath (str): The path to the file containing the embeddings.
                        Assumes the file is a NumPy binary (.npy).

    Returns:
        np.ndarray: An n x k NumPy array of embeddings, or None if loading fails.
    """
    if not os.path.exists(filepath):
        print(f"Error: Input file not found at {filepath}")
        return None

    try:
        # Assuming the embeddings are stored as a NumPy binary file (.npy)
        embeddings = np.load(filepath)
        print(f"Loaded embeddings with shape: {embeddings.shape} from {filepath}")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings from {filepath}: {e}")
        return None

def reduce_embeddings(embeddings, method='tsne', n_components=2, initial_pca_n_components=None, **kwargs):
    """
    Reduces the dimensionality of the embeddings using the specified method.
    Can include an optional initial PCA step.

    Args:
        embeddings (np.ndarray): The input embeddings (n x k).
        method (str): The final dimensionality reduction method ('tsne', 'umap', or 'pca').
        n_components (int): The target dimensionality for the final reduction (usually 2 or 3).
        initial_pca_n_components (int, optional): If specified and greater than
                                                  n_components, perform an initial
                                                  PCA to this many components before
                                                  applying the main method. Defaults to None.
        **kwargs: Additional parameters for the chosen reduction method
                  (e.g., perplexity for t-SNE, n_neighbors for UMAP).

    Returns:
        np.ndarray: The reduced embeddings (n x n_components), or None if method is invalid or an error occurs.
    """
    current_embeddings = embeddings
    original_dim = embeddings.shape[1]

    # Perform initial PCA if requested and if the number of components is less than the original dimensions
    if initial_pca_n_components is not None and original_dim > initial_pca_n_components:
        # Also check if initial_pca_n_components makes sense (e.g., > n_components)
        if initial_pca_n_components >= original_dim or initial_pca_n_components <= n_components:
            print(f"Warning: initial_pca_n_components ({initial_pca_n_components}) must be less than original dimensions ({original_dim}) and typically greater than final n_components ({n_components}). Skipping initial PCA.")
        else:
            print(f"Performing initial PCA to {initial_pca_n_components} components...")
            try:
                # Using a fixed random_state for reproducibility
                pca_initial = PCA(n_components=initial_pca_n_components, random_state=42)
                current_embeddings = pca_initial.fit_transform(embeddings)
                print(f"Initial PCA complete. Embeddings shape: {current_embeddings.shape}")
            except Exception as e:
                print(f"Error during initial PCA: {e}")
                return None

    print(f"Performing final reduction using {method} to {n_components} components...")
    try:
        if method == 'tsne':
            # Set a default random_state for reproducibility
            # Note: TSNE's result can still vary due to its nature
            tsne = TSNE(n_components=n_components, random_state=42, **kwargs)
            reduced_embeddings = tsne.fit_transform(current_embeddings)
        elif method == 'umap':
            # Make sure you have imported umap and it's not None
            if 'umap' not in globals() or umap is None:
                print("UMAP library not found. Please install 'umap-learn'.")
                return None
            # Set a default random_state for reproducibility
            reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
            reduced_embeddings = reducer.fit_transform(current_embeddings)
        elif method == 'pca':
            # Set a default random_state for reproducibility
            pca_final = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = pca_final.fit_transform(current_embeddings)
        else:
            print(f"Unknown final method: {method}. Choose 'tsne', 'umap', or 'pca'.")
            return None
    except Exception as e:
        print(f"Error during final dimensionality reduction with {method}: {e}")
        return None

    print("Final reduction complete.")
    return reduced_embeddings

def plot_combined_embeddings_seaborn(reduced_embeddings1, reduced_embeddings2, title="Combined 2D Visualization", output_path=None):
    """
    Plots the combined 2D reduced embeddings using Seaborn, color-coded, and saves or displays it.

    Args:
        reduced_embeddings1 (np.ndarray): The reduced embeddings for the first dataset (n1 x 2).
        reduced_embeddings2 (np.ndarray): The reduced embeddings for the second dataset (n2 x 2).
        title (str): The title for the plot.
        output_path (str, optional): The path to save the plot image. If None,
                                     the plot is displayed.
    """
    if reduced_embeddings1 is None or reduced_embeddings2 is None:
        print("One or both sets of reduced embeddings are missing.")
        return

    if reduced_embeddings1.shape[1] != 2 or reduced_embeddings2.shape[1] != 2:
        print("Cannot plot. Only 2D embeddings are supported for combined plotting.")
        return

    print("Plotting combined 2D embeddings using Seaborn...")

    plt.figure(figsize=(10, 8))

    df1 = pd.DataFrame(reduced_embeddings1, columns=['Component 1', 'Component 2'])
    df1['Dataset'] = 'Dataset 1'  # Add a column to identify the dataset
    df2 = pd.DataFrame(reduced_embeddings2, columns=['Component 1', 'Component 2'])
    df2['Dataset'] = 'Dataset 2'
    df = pd.concat([df1, df2]) # Combine the dataframes

    sns.scatterplot(
        x='Component 1',
        y='Component 2',
        hue='Dataset',  # Color by the 'Dataset' column
        data=df,
        alpha=0.6,
        s=50
    )
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.grid(True)

    if output_path:
        try:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Combined plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving combined plot to {output_path}: {e}")
    else:
        plt.show()

    plt.close()  # Close the plot figure after displaying or saving
    print("Combined plotting routine finished.")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize combined sample embeddings using dimensionality reduction.')
    parser.add_argument(
        '--input_path1',
        type=str,
        required=True,
        help='Path to the input file containing embeddings for the first dataset (e.g., embeddings1.npy)'
    )
    parser.add_argument(
        '--input_path2',
        type=str,
        required=True,
        help='Path to the input file containing embeddings for the second dataset (e.g., embeddings2.npy)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='Optional path to save the output plot (e.g., combined_visualization.png). If not provided, the plot is displayed.'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='tsne',
        choices=['tsne', 'umap', 'pca'],
        help='Dimensionality reduction method (default: tsne)'
    )
    parser.add_argument(
        '--n_components',
        type=int,
        default=2,
        choices=[2, 3],
        help='Number of components for dimensionality reduction (default: 2)'
    )
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='Perplexity for t-SNE (only applies when method is tsne, default: 30.0)'
    )
    parser.add_argument(
        '--n_neighbors',
        type=int,
        default=15,
        help='Number of neighbors for UMAP (only applies when method is umap, default: 15)'
    )
    parser.add_argument(
        '--min_dist',
        type=float,
        default=0.1,
        help='Minimum distance for UMAP (only applies when method is umap, default: 0.1)'
    )
    parser.add_argument(
        '--initial_pca_n_components',
        type=int,
        help='Optional: Number of components for an initial PCA step before the main reduction.'
    )

    args = parser.parse_args()

    # 1. Load your embeddings from the specified input paths
    embeddings1 = load_embeddings_from_file(args.input_path1)
    embeddings2 = load_embeddings_from_file(args.input_path2)

    if embeddings1 is not None and embeddings2 is not None:
        # 2. Combine the embeddings *before* dimensionality reduction
        combined_embeddings = np.concatenate((embeddings1, embeddings2), axis=0)

        # 3. Reduce dimensionality using the chosen method and parameters
        reduction_kwargs = {}
        if args.method == 'tsne':
            reduction_kwargs['perplexity'] = args.perplexity
        elif args.method == 'umap':
            reduction_kwargs['n_neighbors'] = args.n_neighbors
            reduction_kwargs['min_dist'] = args.min_dist

        reduced_embeddings = reduce_embeddings(
            combined_embeddings,
            method=args.method,
            n_components=args.n_components,
            initial_pca_n_components=args.initial_pca_n_components,
            **reduction_kwargs
        )

        # 4. Split the reduced embeddings back into two groups for plotting
        if reduced_embeddings is not None:
            n1 = embeddings1.shape[0]  # Number of samples in the first dataset
            reduced_embeddings1 = reduced_embeddings[:n1]
            reduced_embeddings2 = reduced_embeddings[n1:]

            # 5. Plot the combined reduced embeddings and save or display
            plot_combined_embeddings_seaborn(
                reduced_embeddings1,
                reduced_embeddings2,
                title=f'{args.method.upper()} Visualization ({args.n_components}D) - Combined Datasets',
                output_path=args.output_path
            )