#!/bin/bash
#!/bin/bash

# Define the base directories
INPUT_BASE_DIR="/mnt/disk3/verl/embedding"
OUTPUT_BASE_DIR="$HOME/verl/results" # Use $HOME for your home directory

# Define the specific embedding subdirectories
# Make sure these paths relative to INPUT_BASE_DIR lead to the embeddings.npy file's parent directory
EMBEDDING_SUBDIRS=(
    "deepscaler/e5-mistral-7b-instruct"
    "openr1-math/e5-mistral-7b-instruct"
)

# Define the name of the embeddings file within each subdirectory
EMBEDDING_FILENAME="embeddings.npy"

# Define the path to your Python script
PYTHON_SCRIPT="./plot_distribution.py" # Make sure this is the correct script name

# Define the dimensionality reduction parameters for UMAP
DR_METHOD="umap"          # Change method to UMAP
DR_N_COMPONENTS=2         # Typically 2 for visualization

# Define initial PCA components (highly recommended for your data size)
# Set to '' or comment out the argument line if you don't want initial PCA
INITIAL_PCA_COMPONENTS=256 # Or 50, 100, etc. Experiment with this.

# Define UMAP parameters to sweep over
N_NEIGHBORS_VALUES=(15 30 50 100) # Add or remove values as needed for n_neighbors
MIN_DIST_VALUES=(0.0 0.1 0.25 0.5)  # Add or remove values as needed for min_dist

# Ensure the Python script is executable (optional, but good practice)
# chmod +x $PYTHON_SCRIPT # Uncomment and run once if needed

echo "--- Starting Batch Visualization with UMAP Parameter Sweep ---"

# Loop through each embedding subdirectory
for SUBDIR in "${EMBEDDING_SUBDIRS[@]}"; do
    INPUT_PATH="${INPUT_BASE_DIR}/${SUBDIR}/${EMBEDDING_FILENAME}"
    OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${SUBDIR}"

    echo "--- Processing embeddings in: ${SUBDIR} ---"
    echo "Input path: ${INPUT_PATH}"
    echo "Output base directory: ${OUTPUT_SUBDIR}"

    # Check if the input file exists
    if [ ! -f "$INPUT_PATH" ]; then
        echo "Error: Input file not found at ${INPUT_PATH}. Skipping subdirectory."
        continue # Skip to the next subdirectory
    fi

    # Create the output subdirectory if it doesn't exist
    mkdir -p "$OUTPUT_SUBDIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory ${OUTPUT_SUBDIR}. Skipping subdirectory."
        continue # Skip to the next subdirectory
    fi

    # Loop through each combination of UMAP parameters
    for N_NEIGHBORS in "${N_NEIGHBORS_VALUES[@]}"; do
        for MIN_DIST in "${MIN_DIST_VALUES[@]}"; do
            # Define the output filename including method, components, n_neighbors, and min_dist
            # Using underscores instead of dots in filename for compatibility
            OUTPUT_PLOT_FILENAME="${DR_METHOD}_${DR_N_COMPONENTS}d_nn${N_NEIGHBORS}_md$(echo "$MIN_DIST" | sed 's/\./p/')_visualization.png"
            OUTPUT_PATH="${OUTPUT_SUBDIR}/${OUTPUT_PLOT_FILENAME}"

            echo "  Processing with n_neighbors=${N_NEIGHBORS}, min_dist=${MIN_DIST}"
            echo "  Output plot path: ${OUTPUT_PATH}"

            # Run the Python script
            PYTHON_COMMAND="python \"$PYTHON_SCRIPT\" \
                --input_path \"$INPUT_PATH\" \
                --output_path \"$OUTPUT_PATH\" \
                --method \"$DR_METHOD\" \
                --n_components \"$DR_N_COMPONENTS\" \
                --n_neighbors \"$N_NEIGHBORS\" \
                --min_dist \"$MIN_DIST\""

            # Add initial PCA argument if specified
            if [ -n "$INITIAL_PCA_COMPONENTS" ]; then
                 PYTHON_COMMAND="$PYTHON_COMMAND --initial_pca_n_components \"$INITIAL_PCA_COMPONENTS\""
            fi

            # Execute the command
            eval "$PYTHON_COMMAND"

            # Check the exit status of the python script
            if [ $? -eq 0 ]; then
                echo "  Successfully generated plot for nn=${N_NEIGHBORS}, md=${MIN_DIST}"
            else
                echo "  Error generating plot for nn=${N_NEIGHBORS}, md=${MIN_DIST}"
            fi
            echo "" # Add a newline for readability between runs
        done
    done

    echo "--- Finished processing subdirectory: ${SUBDIR} ---"
    echo "" # Add a newline for readability between subdirectories

done

echo "--- Batch processing finished ---"