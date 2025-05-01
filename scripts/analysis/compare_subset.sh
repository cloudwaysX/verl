#!/bin/bash

# Define the path to your Python script
# Make sure this path is correct relative to where you save the shell script,
# or use the absolute path.
ANALYZE_SCRIPT="./analyze_common_text.py"

# Define the three specific files involved in these comparisons
FILE_1="/home/yifangc_google_com/verl/results/deepscaler/e5-mistral-7b-instruct/oed_DeepSeek-R1-Distill-Qwen-1.5B_1024/orderd_coreset_idxs.npy"
FILE_2="/home/yifangc_google_com/verl/results/deepscaler/e5-mistral-7b-instruct/oed_DeepSeek-R1-Distill-Qwen-1.5B_1024/orderd_reversed_coreset_initsize1_idxs.npy"
FILE_3="/home/yifangc_google_com/verl/results/deepscaler/e5-mistral-7b-instruct/oed_DeepSeek-R1-Distill-Qwen-1.5B_1024/orderd_reversed_coreset_initsize100_idxs_42.npy"

# Define the specific pairs of files to compare
# Each element in this array is a string containing the two file paths separated by a space.
# We define the pairs: (File 1, File 2), (File 2, File 3), and (File 3, File 1).
COMPARISON_PAIRS=(
    "$FILE_1 $FILE_2"
    "$FILE_2 $FILE_3"
    "$FILE_3 $FILE_1"
)

# Define a large size to effectively compare the full length of the shorter list.
# This number should be larger than the expected maximum possible size of your lists.
FULL_SIZE_INDICATOR=999999999

echo "Starting specific pairwise comparison process..."

# Loop through each defined pair of file paths
for pair in "${COMPARISON_PAIRS[@]}"; do
    # Read the two file paths from the current pair string into separate variables
    read file1 file2 <<< "$pair"

    echo "--------------------------------------------------"
    echo "Comparing:"
    echo "  File 1: $file1"
    echo "  File 2: $file2"
    echo "--------------------------------------------------"

    # Check if both files in the current pair exist before attempting the comparison
    if [ ! -f "$file1" ]; then
        echo "Error: File not found at '$file1'. Skipping this comparison."
        continue # Skip to the next pair in the loop
    fi
    if [ ! -f "$file2" ]; then
        echo "Error: File not found at '$file2'. Skipping this comparison."
        continue # Skip to the next pair in the loop
    fi

    # Execute the Python script with the two files and the large size.
    # The Python script will calculate the unique common elements
    # considering the full length of the shorter file due to the large size parameter.
    # Its output will include the filenames as we modified it to do previously.
    python "$ANALYZE_SCRIPT" "$file1" "$file2" 

    echo "" # Add a blank line for separation between the output of different comparisons
done

echo "Specific pairwise comparison process complete."
echo "--------------------------------------------------"