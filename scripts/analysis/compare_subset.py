import numpy as np
import sys

def count_same_elements_at_common_indices(npy_file1, npy_file2):
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
        list2 = np.load(npy_file2)
    except FileNotFoundError:
        print(f"Error: One or both of the files ({npy_file1}, {npy_file2}) not found.")
        return -1
    except Exception as e:
        print(f"Error reading .npy files: {e}")
        return -1

    same_count = 0
    # Iterate through both lists simultaneously using zip.
    # zip stops when the shortest list is exhausted, naturally handling
    # the "indices contained in both" requirement.
    for item1, item2 in zip(list1, list2):
        if item1 == item2:
            same_count += 1

    return same_count

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script_name.py <npy_file1> <npy_file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    num_same = count_same_elements_at_common_indices(file1, file2)

    if num_same != -1:
        print(f"Number of indices contained in both files with the same element: {num_same}")