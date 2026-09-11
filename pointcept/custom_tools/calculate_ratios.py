import os
import glob
import numpy as np

def calculate_class_ratios(data_root):
    # Class index mapping based on your dataset
    class_names = {0: "trunk", 1: "branch", 2: "wire"}
    class_counts = {0: 0, 1: 0, 2: 0}
    ignored_points = 0

    print(f"Scanning directory: {data_root}...")

    # Find all segment.npy files recursively (handles train/val and tree subfolders)
    search_pattern = os.path.join(data_root, '**', '*segment.npy')
    segment_files = glob.glob(search_pattern, recursive=True)

    if not segment_files:
        print(f"Error: No '*segment.npy' files found in {data_root}")
        return

    print(f"Found {len(segment_files)} point cloud segment files. Tallying points...\n")

    for file_path in segment_files:
        # Load the numpy array
        labels = np.load(file_path).astype(np.int64)
        
        # Count occurrences of each unique value in the array
        unique, counts = np.unique(labels, return_counts=True)
        
        for val, count in zip(unique, counts):
            if val in class_counts:
                class_counts[val] += count
            elif val == -1:
                ignored_points += count
            else:
                print(f"Warning: Unexpected class label '{val}' found in {file_path}")

    # --- Calculations ---
    total_valid_points = sum(class_counts.values())
    
    if total_valid_points == 0:
        print("Error: No valid labeled points found.")
        return

    print("-" * 50)
    print("DATASET DISTRIBUTION RATIOS")
    print("-" * 50)
    print(f"Total Valid Points: {total_valid_points:,}")
    if ignored_points > 0:
        print(f"Ignored Points (-1): {ignored_points:,}")
    print()

    for cls_idx in range(len(class_names)):
        name = class_names[cls_idx]
        count = class_counts[cls_idx]
        ratio = (count / total_valid_points) * 100
        print(f"Class {cls_idx} ({name:<8}): {count:>15,} points  |  {ratio:>6.2f}%")

    print("\n" + "-" * 50)
    print("FOCAL LOSS ALPHA CALCULATION (Inverse Frequency)")
    print("-" * 50)
    
    # Check for missing classes to prevent division by zero
    if any(c == 0 for c in class_counts.values()):
        print("Warning: One or more classes have 0 points. Cannot calculate alpha weights safely.")
    else:
        # 1. Calculate inverse of the counts
        inverses = [1.0 / class_counts[i] for i in range(len(class_names))]
        # 2. Normalize so the alpha values sum to 1.0
        sum_inverses = sum(inverses)
        alpha_weights = [inv / sum_inverses for inv in inverses]
        
        print(f"alpha=[{alpha_weights[0]:.6f}, {alpha_weights[1]:.6f}, {alpha_weights[2]:.6f}]")

if __name__ == "__main__":
    # Adjust this path if you are running it from a different directory
    DATA_ROOT = "data/BRANCH_v2/numpy_dataset/partial"
    
    calculate_class_ratios(DATA_ROOT)