#!/bin/bash

# Define the path to your Python script that compares TWO files
# Make sure this path is correct relative to where you save the shell script,
# or use the absolute path.
ANALYZE_SCRIPT="./analyze_common_text.py"

# Define a large size to effectively compare the full length of the shorter list.
# This number should be larger than the expected maximum possible size of your lists.
FULL_SIZE_INDICATOR=999999999

# Define the base directory
HOME_DIR="/home/yifangc_google_com/verl/results/deepscaler/"

# Define the specific files for the three pairs using the base directory
# First pair files
FILE_A_1="${HOME_DIR}/e5-mistral-7b-instruct/trainmodel_agnostic/redant_idxs_size4030.json"
FILE_A_2="${HOME_DIR}/gecko_en_1b_tpu/trainmodel_agnostic/redant_idxs_size4030.json"

# Second pair files
FILE_B_1="${HOME_DIR}/e5-mistral-7b-instruct/trainmodel_agnostic/redant_idxs_size1007.json"
FILE_B_2="${HOME_DIR}/gecko_en_1b_tpu/trainmodel_agnostic/redant_idxs_size1007.json"

# Third pair files
FILE_C_1="${HOME_DIR}/e5-mistral-7b-instruct/trainmodel_agnostic/redant_idxs_size10078.json"
FILE_C_2="${HOME_DIR}/gecko_en_1b_tpu/trainmodel_agnostic/redant_idxs_size10078.json"

# Define the specific pairs of files to compare
# Each element in this array is a string containing the two file paths separated by a space.
COMPARISON_PAIRS=(
    "$FILE_A_1 $FILE_A_2"
    "$FILE_B_1 $FILE_B_2"
    "$FILE_C_1 $FILE_C_2"
)

echo "Starting specific pairwise comparison process for the provided pairs..."

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

    # Execute the Python script (the one for two files) with the current pair and the large size.
    # The Python script will calculate the unique common elements
    # considering the full length of the shorter file due to the large size parameter,
    # and its output will include the filenames.
    python "$ANALYZE_SCRIPT" "$file1" "$file2" "$FULL_SIZE_INDICATOR"

    echo "" # Add a blank line for separation between the output of different comparisons
done

echo "Specific pairwise comparison process complete."
echo "--------------------------------------------------"