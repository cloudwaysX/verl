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

# Define the desired output filename for the plot
OUTPUT_PLOT_FILENAME="tsne_2d_visualization.png" # You can change this filename

# Define the path to your Python script
PYTHON_SCRIPT="./visualize_embeddings.py" # Adjust if your script is elsewhere

# Define the dimensionality reduction parameters (you can change these)
DR_METHOD="tsne"
DR_N_COMPONENTS=2
TSNE_PERPLEXITY=30

# Ensure the Python script is executable (optional, but good practice)
# chmod +x $PYTHON_SCRIPT # Uncomment and run once if needed

# Loop through each embedding subdirectory
for SUBDIR in "${EMBEDDING_SUBDIRS[@]}"; do
    INPUT_PATH="${INPUT_BASE_DIR}/${SUBDIR}/${EMBEDDING_FILENAME}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SUBDIR}"
    OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_PLOT_FILENAME}"

    echo "--- Processing embeddings in: ${SUBDIR} ---"
    echo "Input path: ${INPUT_PATH}"
    echo "Output path: ${OUTPUT_PATH}"

    # Check if the input file exists
    if [ ! -f "$INPUT_PATH" ]; then
        echo "Error: Input file not found at ${INPUT_PATH}. Skipping."
        continue # Skip to the next subdirectory
    fi

    # Create the output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory ${OUTPUT_DIR}. Skipping."
        continue # Skip to the next subdirectory
    fi

    # Run the Python script
    python "$PYTHON_SCRIPT" \
        --input_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --method "$DR_METHOD" \
        --n_components "$DR_N_COMPONENTS" \
        --perplexity "$TSNE_PERPLEXITY" # Only used if method is tsne

    # Check the exit status of the python script
    if [ $? -eq 0 ]; then
        echo "Successfully processed ${SUBDIR}"
    else
        echo "Error processing ${SUBDIR}"
    fi

    echo "" # Add a newline for readability between runs
done

echo "--- Batch processing finished ---"