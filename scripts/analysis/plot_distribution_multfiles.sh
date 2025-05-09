#!/bin/bash
#!/bin/bash

# --- Define your specific input file paths ---
INPUT_PATH1="/mnt/disk3/verl/embedding/deepscaler/e5-mistral-7b-instruct/embeddings.npy"
INPUT_PATH2="/mnt/disk3/verl/embedding/openr1-math/e5-mistral-7b-instruct/embeddings.npy"

# --- Define output base directory for combined plots ---
# This will create a new directory like ~/verl/results/combined_plots/
OUTPUT_BASE_DIR="$HOME/verl/results/combined_plots"

# --- Define the path to your Python script ---
PYTHON_SCRIPT="./plot_distribution_multfiles.py" # Make sure this is the correct script name and path

# --- Dimensionality Reduction Method and Components ---
DR_METHOD="umap"          # Set to 'umap' for UMAP sweep, or 'tsne' for t-SNE sweep
DR_N_COMPONENTS=2         # Keep at 2 for 2D visualization of combined data

# --- Initial PCA Components (Highly recommended for your data scale) ---
# Set to '' (empty string) to disable initial PCA if your Python script
# has been modified to handle it as an optional argument.
INITIAL_PCA_COMPONENTS=256 # Or 50, 100, etc. Experiment with this.

# --- Define parameters for UMAP sweep ---
# These will be used if DR_METHOD is 'umap'
UMAP_N_NEIGHBORS_VALUES=(15 30 50 100) # Add or remove values as needed for n_neighbors
UMAP_MIN_DIST_VALUES=(0.0 0.1 0.25 0.5)  # Add or remove values as needed for min_dist

# --- Define parameters for t-SNE sweep (if you switch DR_METHOD to 'tsne') ---
# These will be used if DR_METHOD is 'tsne'
TSNE_PERPLEXITY_VALUES=(5 10 30 50 100) # Add or remove values as needed for perplexity

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- You typically don't need to modify anything below this line ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

echo "--- Starting Combined Visualization Batch Processing ---"
echo "Input Dataset 1: ${INPUT_PATH1}"
echo "Input Dataset 2: ${INPUT_PATH2}"
echo "Output Base Directory: ${OUTPUT_BASE_DIR}"
echo "Dimensionality Reduction Method: ${DR_METHOD}"
echo "Target Components: ${DR_N_COMPONENTS}D"

# Check if input files exist
if [ ! -f "$INPUT_PATH1" ]; then
    echo "Error: Input file 1 not found at ${INPUT_PATH1}. Exiting."
    exit 1
fi
if [ ! -f "$INPUT_PATH2" ]; then
    echo "Error: Input file 2 not found at ${INPUT_PATH2}. Exiting."
    exit 1
fi

# Create the output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create output directory ${OUTPUT_BASE_DIR}. Exiting."
    exit 1
fi

# --- Execute based on the chosen method ---
if [ "$DR_METHOD" == "umap" ]; then
    echo "Sweeping UMAP parameters..."
    # Loop through each combination of UMAP parameters
    for N_NEIGHBORS in "${UMAP_N_NEIGHBORS_VALUES[@]}"; do
        for MIN_DIST in "${UMAP_MIN_DIST_VALUES[@]}"; do
            # Define the output filename including method, components, n_neighbors, and min_dist
            # Using 'p' for dot in filename for compatibility (e.g., 0.1 -> 0p1)
            OUTPUT_PLOT_FILENAME="${DR_METHOD}_${DR_N_COMPONENTS}d_nn${N_NEIGHBORS}_md$(echo "$MIN_DIST" | sed 's/\./p/')_combined.png"
            OUTPUT_PATH="${OUTPUT_BASE_DIR}/${OUTPUT_PLOT_FILENAME}"

            echo "  Processing with n_neighbors=${N_NEIGHBORS}, min_dist=${MIN_DIST}"
            echo "  Output plot path: ${OUTPUT_PATH}"

            # Construct the Python command
            PYTHON_COMMAND="python \"$PYTHON_SCRIPT\" \
                --input_path1 \"$INPUT_PATH1\" \
                --input_path2 \"$INPUT_PATH2\" \
                --output_path \"$OUTPUT_PATH\" \
                --method \"$DR_METHOD\" \
                --n_components \"$DR_N_COMPONENTS\" \
                --n_neighbors \"$N_NEIGHBORS\" \
                --min_dist \"$MIN_DIST\""

            # Add initial PCA argument if specified (not empty)
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

elif [ "$DR_METHOD" == "tsne" ]; then
    echo "Sweeping t-SNE perplexity parameters..."
    # Loop through each perplexity value
    for PERPLEXITY in "${TSNE_PERPLEXITY_VALUES[@]}"; do
        # Define the output filename including method, components, and perplexity
        OUTPUT_PLOT_FILENAME="${DR_METHOD}_${DR_N_COMPONENTS}d_p${PERPLEXITY}_combined.png"
        OUTPUT_PATH="${OUTPUT_BASE_DIR}/${OUTPUT_PLOT_FILENAME}"

        echo "  Processing with perplexity: ${PERPLEXITY}"
        echo "  Output plot path: ${OUTPUT_PATH}"

        # Construct the Python command
        PYTHON_COMMAND="python \"$PYTHON_SCRIPT\" \
            --input_path1 \"$INPUT_PATH1\" \
            --input_path2 \"$INPUT_PATH2\" \
            --output_path \"$OUTPUT_PATH\" \
            --method \"$DR_METHOD\" \
            --n_components \"$DR_N_COMPONENTS\" \
            --perplexity \"$PERPLEXITY\""

        # Add initial PCA argument if specified (not empty)
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
        echo "" # Add a newline for readability between runs
    done

else
    echo "Error: Unknown DR_METHOD specified. Please choose 'umap' or 'tsne'."
    exit 1
fi


echo "--- Batch processing finished ---"