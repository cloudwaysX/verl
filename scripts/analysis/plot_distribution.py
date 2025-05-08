import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import argparse
import os

# Optional: Import UMAP if you have it installed
# try:
#     import umap
# except ImportError:
#     umap = None
#     print("UMAP not installed. Install with 'pip install umap-learn' to use it.")


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

def reduce_embeddings(embeddings, method='tsne', n_components=2, **kwargs):
    """
    Reduces the dimensionality of the embeddings using the specified method.

    Args:
        embeddings (np.ndarray): The input embeddings (n x k).
        method (str): The dimensionality reduction method ('tsne', 'umap', or 'pca').
        n_components (int): The target dimensionality (usually 2 or 3).
        **kwargs: Additional parameters for the chosen reduction method
                  (e.g., perplexity for t-SNE, n_neighbors for UMAP).

    Returns:
        np.ndarray: The reduced embeddings (n x n_components), or None if method is invalid or an error occurs.
    """
    print(f"Reducing embeddings using {method} to {n_components} components...")
    try:
        if method == 'tsne':
            # Set a default random_state for reproducibility
            tsne = TSNE(n_components=n_components, random_state=42, **kwargs)
            reduced_embeddings = tsne.fit_transform(embeddings)
        elif method == 'umap':
            if umap is None:
                print("UMAP library not found. Please install 'umap-learn'.")
                return None
            # Set a default random_state for reproducibility
            reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
            reduced_embeddings = reducer.fit_transform(embeddings)
        elif method == 'pca':
             # Set a default random_state for reproducibility
            pca = PCA(n_components=n_components, random_state=42, **kwargs)
            reduced_embeddings = pca.fit_transform(embeddings)
        else:
            print(f"Unknown method: {method}. Choose 'tsne', 'umap', or 'pca'.")
            return None
    except Exception as e:
        print(f"Error during dimensionality reduction with {method}: {e}")
        return None


    print("Reduction complete.")
    return reduced_embeddings

def plot_reduced_embeddings_seaborn(reduced_embeddings, title="2D Visualization of Embeddings", output_path=None):
    """
    Plots the 2D or 3D reduced embeddings using Seaborn and saves or displays it.

    Args:
        reduced_embeddings (np.ndarray): The reduced embeddings (n x 2 or n x 3).
        title (str): The title for the plot.
        output_path (str, optional): The path to save the plot image. If None,
                                     the plot is displayed.
    """
    if reduced_embeddings is None:
        print("No reduced embeddings to plot.")
        return

    n_components = reduced_embeddings.shape[1]

    if n_components not in [2, 3]:
        print(f"Cannot plot {n_components}-dimensional embeddings. Only 2D and 3D supported.")
        return

    print(f"Plotting {n_components}D embeddings using Seaborn...")

    plt.figure(figsize=(10, 8))

    if n_components == 2:
        df = pd.DataFrame(reduced_embeddings, columns=['Component 1', 'Component 2'])
        sns.scatterplot(
            x='Component 1',
            y='Component 2',
            data=df,
            alpha=0.6,
            s=50
        )
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif n_components == 3:
        # Using Matplotlib's 3D plotting for 3 components
        print("Using Matplotlib for 3D plotting as Seaborn's scatterplot is 2D.")
        ax = plt.gcf().add_subplot(111, projection='3d')
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], alpha=0.6, s=50)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

    plt.title(title)
    plt.grid(True)

    if output_path:
        try:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot to {output_path}: {e}")
    else:
        plt.show()

    plt.close() # Close the plot figure after displaying or saving
    print("Plotting routine finished.")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize sample embeddings using dimensionality reduction.')
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to the input file containing sample embeddings (e.g., embeddings.npy)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='Optional path to save the output plot (e.g., visualization.png). If not provided, the plot is displayed.'
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
    # Add more arguments for other method parameters if needed (e.g., n_neighbors for UMAP)

    args = parser.parse_args()

    # 1. Load your embeddings from the specified input path
    my_embeddings = load_embeddings_from_file(args.input_path)

    if my_embeddings is not None:
        # 2. Reduce dimensionality using the chosen method and parameters
        reduction_kwargs = {}
        if args.method == 'tsne':
            reduction_kwargs['perplexity'] = args.perplexity
            # Add other t-SNE specific args here if you add them to the parser
        # elif args.method == 'umap' and umap is not None:
            # Add UMAP specific args here if you add them to the parser

        reduced_embeddings = reduce_embeddings(
            my_embeddings,
            method=args.method,
            n_components=args.n_components,
            **reduction_kwargs
        )

        # 3. Plot the reduced embeddings and save or display
        if reduced_embeddings is not None:
            plot_reduced_embeddings_seaborn(
                reduced_embeddings,
                title=f'{args.method.upper()} Visualization ({args.n_components}D)',
                output_path=args.output_path
            )