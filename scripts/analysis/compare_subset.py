import numpy as np
import json
import sys
import os # Import os module to check file extension

def load_data(filepath):
    """Loads data from a .npy or .json file."""
    _, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()

    if file_extension == '.npy':
        try:
            return np.load(filepath)
        except Exception as e:
            print(f"Error reading .npy file {filepath}: {e}")
            return None
    elif file_extension == '.json':
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    print(f"Error: JSON file {filepath} does not contain a list at the top level.")
                    return None
        except FileNotFoundError:
             print(f"Error: File not found at {filepath}")
             return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filepath}: {e}")
            return None
        except Exception as e:
            print(f"Error reading .json file {filepath}: {e}")
            return None
    else:
        print(f"Error: Unsupported file extension for {filepath}. Please use .npy or .json.")
        return None

def count_unique_common_elements_in_slice(list1, list2, size):
    """
    Takes two lists and a size, and counts the number of unique elements
    common to the first 'size' elements of each list.
    Assumes "common" refers to unique values present in both slices.
    Returns count or None if size is invalid, or -1 for processing errors.
    """
    if size < 0:
        # A negative size is not meaningful for "first n samples".
        return None # Indicate invalid size

    # Apply slicing
    # Slicing will take up to 'size' elements.
    # If the list is shorter than 'size', it takes all elements up to the end.
    list1_sliced = list1[:size]
    list2_sliced = list2[:size]

    # Convert sliced lists to sets to easily find unique common elements
    try:
        set1 = set(list1_sliced)
        set2 = set(list2_sliced)
    except TypeError as e:
        # Handle cases where list elements are not hashable (cannot be put in a set)
        print(f"Error converting list elements to set for size {size}. Ensure list elements are hashable: {e}")
        return -1 # Indicate a processing error for this specific size


    # Find the intersection of the two sets (unique common elements)
    common_elements = set1.intersection(set2)

    # Compute the number of unique common elements
    count_of_common = len(common_elements)

    return count_of_common


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python your_script_name.py <file1_path> <file2_path> [step_size] [start_size]")
        print("Supported file types: .npy, .json")
        print("Provide two filenames. Optional arguments: step_size (int, default 100), start_size (int, default 1).")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # Default parameters for analysis range
    step_size = 1000
    start_size = 1000

    # Parse optional arguments for step_size and start_size
    if len(sys.argv) > 3:
        try:
            step_size = int(sys.argv[3])
            if step_size <= 0:
                 print("Error: Step size must be a positive integer.")
                 sys.exit(1)
        except ValueError:
            print(f"Error: Invalid step size '{sys.argv[3]}'. Step size must be an integer.")
            sys.exit(1)


    # Load the complete lists from the files once
    full_list1 = load_data(file1)
    full_list2 = load_data(file2)

    # Check if data loading was successful for both files
    if full_list1 is None or full_list2 is None:
        sys.exit(1) # Error message was already printed by load_data

    # Determine the maximum size for analysis (length of the shorter list)
    max_size = min(len(full_list1), len(full_list2))

    # Adjust start_size if it's out of the valid range [0, max_size].
    start_size = 0


    # Generate a range of sizes for analysis
    # We want to include sizes from start_size up to max_size with a given step.
    # Ensure max_size is always included in the points, unless start_size > max_size.
    sizes_to_analyze = list(range(start_size, max_size + 1, step_size))
    # Add max_size to the list if it's not already included and is reachable.
    if max_size not in sizes_to_analyze and max_size >= start_size:
         sizes_to_analyze.append(max_size)

    # Sort sizes for a clear, ordered output
    sizes_to_analyze.sort()

    print(f"Analyzing unique common elements for increasing slice sizes...")
    print(f"Files: {file1}, {file2}")
    print(f"Analysis range: from {start_size} to {max_size} with step {step_size}")
    print("-" * 40) # Separator for clarity
    print(f"{'Slice Size':<15} | {'Unique Common Elements':<25}")
    print("-" * 40)

    # Perform calculations for each size and print results
    for current_size in sizes_to_analyze:
        # Skip sizes larger than max_size (should be handled by list generation, but defensive)
        if current_size > max_size or current_size==0:
             continue

        count = count_unique_common_elements_in_slice(full_list1, full_list2, current_size)

        # Print the result if the count is a valid number
        if isinstance(count, (int, float)) and count != -1:
             print(f"{current_size:<15} | {count:<25}")
        # If count was None or -1, a warning or error was already printed by the helper function.

    print("-" * 40) # End separator