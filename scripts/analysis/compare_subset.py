import numpy as np
import sys

def count_same_elements_at_common_indices(npy_file1, npy_file2, size):
    """
    Reads two lists from .npy files and counts how many elements are the same
    at the indices that are common to both lists (up to the length of the shorter list).

    Args:
        npy_file1 (str): The path to the first .npy file.
        npy_file2 (str): The path to the second .npy file.

    Returns:
        int: The number of common indices where the elements in both lists are the same.
             Returns -1 if the files cannot be read.
    """
    try:
        list1 = np.load(npy_file1)
        list1 = list1[:size]
        list2 = np.load(npy_file2)
        list2 = list2[:size]
    except FileNotFoundError:
        print(f"Error: One or both of the files ({npy_file1}, {npy_file2}) not found.")
        return -1
    except Exception as e:
        print(f"Error reading .npy files: {e}")
        return -1

    # Convert lists to sets to easily find common elements (intersection)
    set1 = set(list1)
    set2 = set(list2)
    print("list 1 has len", len(set1))
    print("list 2 has len", len(set2))
    # Find the intersection of the two sets
    common_elements = set1.intersection(set2)

    # Compute the sum of the common elements
    sum_of_common = len(common_elements)

    return sum_of_common

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python your_script_name.py <npy_file1> <npy_file2> <size>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    size = sys.argv[3]

    num_same = count_same_elements_at_common_indices(file1, file2, int(size))

    if num_same != -1:
        print(f"Number of indices contained in both files with the same element: {num_same}")
