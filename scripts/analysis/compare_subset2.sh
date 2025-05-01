#!/bin/bash

# Define the path to your Python script
# Make sure this path is correct relative to where you save the shell script,
# or use the absolute path.
ANALYZE_SCRIPT="./compare_subset.py"

# Define the four specific files you want to use
FILE_1="/home/yifangc_google_com/verl/results/deepscaler/e5-mistral-7b-instruct/oed_DeepSeek-R1-Distill-Qwen-1.5B_1024/orderd_coreset_idxs.npy"
FILE_2="/home/yifangc_google_com/verl/results/deepscaler/e5-mistral-7b-instruct/oed_DeepSeek-R1-Distill-Qwen-1.5B_1024/orderd_reversed_coreset_initsize1_idxs.npy"
FILE_3="/home/yifangc_google_com/verl/results/deepscaler/e5-mistral-7b-instruct/oed_DeepSeek-R1-Distill-Qwen-1.5B_1024/orderd_reversed_coreset_initsize100_idxs_42.npy"
REFERENCE_FILE="/home/yifangc_google_com/verl/results/deepscaler/e5-mistral-7b-instruct/trainmodel_agnostic/redant_idxs_size4030.json"

# Define the files to compare against the reference file
FILES_TO_COMPARE=(
    "$FILE_1"
    "$FILE_2"
    "$FILE_3"
)

# Define a large size to effectively compare the full length of the shorter list.
# This number should be larger than the expected maximum possible size of your lists.
FULL_SIZE_INDICATOR=999999999

echo "Starting specific file comparison process..."
echo "Reference file for comparison: $REFERENCE_FILE"

# Check if the reference file exists before starting the loop
if [ ! -f "$REFERENCE_FILE" ]; then
    echo "Error: Reference file not found at '$REFERENCE_FILE'. Cannot proceed with any comparisons."
    exit 1 # Exit the script if the reference file is missing
fi


# Loop through the files you want to compare against the reference
for current_file in "${FILES_TO_COMPARE[@]}"; do
    echo "--------------------------------------------------"
    echo "Preparing comparison:"
    echo "  File 1: $current_file"
    echo "  File 2 (Reference): $REFERENCE_FILE"
    echo "--------------------------------------------------"

    # Check if the current file exists before attempting to compare
    if [ ! -f "$current_file" ]; then
        echo "Error: File not found at '$current_file'. Skipping this comparison."
        continue # Skip to the next file in the list
    fi

    # Execute the Python script with the current file, the reference, and the large size.
    # The Python script will calculate the unique common elements
    # considering the full length of the shorter file due to the large size,
    # and its output will include the filenames.
    python "$ANALYZE_SCRIPT" "$current_file" "$REFERENCE_FILE" "$FULL_SIZE_INDICATOR"

    echo "" # Add a blank line for separation between the output of different comparisons
done

echo "Specific file comparison process complete."
echo "--------------------------------------------------"
