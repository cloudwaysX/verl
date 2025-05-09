#!/bin/bash

# Define the base directories
INPUT_BASE_DIR="/mnt/disk3/verl/embedding"
OUTPUT_BASE_DIR="$HOME/verl/results" # Use $HOME for your home directory

# Define the specific embedding subdirectories
# Make sure these paths relative to INPUT_BASE_DIR lead to the embeddings.npy file's parent directory
EMBEDDING_SUBDIRS=(
    # "deepscaler/e5-mistral-7b-instruct"
    "openr1-math/e5-mistral-7b-instruct"
)

# Define the name of the embeddings file within each subdirectory
EMBEDDING_FILENAME="embeddings.npy"

# Define the path to your Python script
PYTHON_SCRIPT="./plot_distribution.py" # Adjust if your script is elsewhere

# Define the dimensionality reduction parameters
DR_METHOD="tsne"         # We are sweeping perplexity for t-SNE
DR_N_COMPONENTS=2        # Typically 2 for visualization

# Define initial PCA components (highly recommended for your data size)
# Set to '' or comment out the argument line if you don't want initial PCA
INITIAL_PCA_COMPONENTS=256 # Or 50, 100, etc. Experiment with this.

# Define the perplexity values to sweep over
PERPLEXITY_VALUES=(5 10 30 50 100) # Add or remove values as needed

# Ensure the Python script is executable (optional, but good practice)
# chmod +x $PYTHON_SCRIPT # Uncomment and run once if needed

echo "--- Starting Batch Visualization with Perplexity Sweep ---"

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

    # Loop through each perplexity value
    for PERPLEXITY in "${PERPLEXITY_VALUES[@]}"; do
        # Define the output filename including method, components, and perplexity
        OUTPUT_PLOT_FILENAME="${DR_METHOD}_${DR_N_COMPONENTS}d_p${PERPLEXITY}_visualization.png"
        OUTPUT_PATH="${OUTPUT_SUBDIR}/${OUTPUT_PLOT_FILENAME}"

        echo "  Processing with perplexity: ${PERPLEXITY}"
        echo "  Output plot path: ${OUTPUT_PATH}"

        # Run the Python script
        # We pass parameters based on the shell script variables and the current loop value
        PYTHON_COMMAND="python \"$PYTHON_SCRIPT\" \
            --input_path \"$INPUT_PATH\" \
            --output_path \"$OUTPUT_PATH\" \
            --method \"$DR_METHOD\" \
            --n_components \"$DR_N_COMPONENTS\" \
            --perplexity \"$PERPLEXITY\""

        # Add initial PCA argument if specified
        if [ -n "$INITIAL_PCA_COMPONENTS" ]; then
             PYTHON_COMMAND="$PYTHON_COMMAND --initial_pca_n_components \"$INITIAL_PCA_COMPONENTS\""
        fi

        # Execute the command
        eval "$PYTHON_COMMAND"

        # Check the exit status of the python script
        if [ $? -eq 0 ]; then
            echo "  Successfully generated plot for perplexity ${PERPLEXITY}"
        else
            echo "  Error generating plot for perplexity ${PERPLEXITY}"
        fi
        echo "" # Add a newline for readability between perplexities
    done

    echo "--- Finished processing subdirectory: ${SUBDIR} ---"
    echo "" # Add a newline for readability between subdirectories

done

echo "--- Batch processing finished ---"
