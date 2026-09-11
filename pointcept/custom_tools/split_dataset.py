"""
Tree-level train/test split for Pointcept point cloud dataset.

Naming conventions:
  Partial views:        tree_1_V_XXXX_YY       (e.g. tree_1_V_0000_0, tree_1_V_0000_1)
  Distinctive merged:  tree_1_V_XXXX_teaser    (e.g. tree_1_V_0000_teaser)

Unique tree identity = XXXX (4-digit number after _V_).

Split modes:
  "partial"     — split only from PARTIAL_DIR
  "merged"      — split only from MERGED_DIR
  "combined"    — split trees from both folders together;
                  same XXXX = same tree, so they never end up on different sides.
"""

import os
import re
import shutil
import random
from collections import defaultdict


# =============================================================================
# CONFIGURATION — edit these, no CLI arguments needed
# =============================================================================

PARTIAL_DIR  = "data/BRANCH_v2/BRANCH_labeled/npy/Partial"            # folder with _YY views
MERGED_DIR   = "data/BRANCH_v2/BRANCH_labeled/npy/Distinctive_merged"  # folder with _teaser views
OUTPUT_DIR   = "data/BRANCH_v2/numpy_dataset/partial"        # where train/ and test/ are written

# "partial" | "merged" | "combined"
MODE         = "partial"

TRAIN_RATIO  = 0.8
SEED         = 42
DRY_RUN      = False   # True = print plan only, no files copied

# =============================================================================


def extract_xxxx(folder_name):
    """
    Extract the 4-digit tree ID (XXXX) from a folder name.
    Works for both:
      tree_1_V_0000_0       -> '0000'
      tree_1_V_0000_teaser  -> '0000'
    """
    match = re.search(r'_V_(\d{4})_', folder_name)
    if match:
        return match.group(1)
    raise ValueError(
        f"Could not extract 4-digit tree ID from: '{folder_name}'\n"
        f"Expected pattern: tree_1_V_XXXX_<something>"
    )


def collect_folders(directory):
    """Return sorted list of subfolders in directory that contain at least one .npy file."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    folders = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if os.path.isdir(full) and any(f.endswith(".npy") for f in os.listdir(full)):
            folders.append(name)
    if not folders:
        raise RuntimeError(f"No subfolders with .npy files found in: {directory}")
    return folders


def build_tree_map(folders, source_dir):
    """Returns dict: xxxx -> [(folder_name, source_dir), ...]"""
    tree_map = defaultdict(list)
    for folder in folders:
        xxxx = extract_xxxx(folder)
        tree_map[xxxx].append((folder, source_dir))
    return tree_map


def merge_tree_maps(*maps):
    """Merge multiple tree maps; same XXXX buckets are combined."""
    combined = defaultdict(list)
    for m in maps:
        for xxxx, entries in m.items():
            combined[xxxx].extend(entries)
    return combined


def perform_split(tree_map, output_dir, train_ratio, seed, dry_run):
    train_dir = os.path.join(output_dir, "train")
    test_dir  = os.path.join(output_dir, "test")

    tree_ids = sorted(tree_map.keys())
    n_trees  = len(tree_ids)
    n_train  = max(1, round(n_trees * train_ratio))
    n_test   = n_trees - n_train

    if n_test == 0:
        raise RuntimeError(
            f"Only {n_trees} unique trees found — too few for a test set "
            f"at {train_ratio:.0%} train ratio."
        )

    random.seed(seed)
    shuffled = tree_ids[:]
    random.shuffle(shuffled)

    train_trees = set(shuffled[:n_train])
    test_trees  = set(shuffled[n_train:])

    train_entries = [(f, d) for tid in train_trees for (f, d) in tree_map[tid]]
    test_entries  = [(f, d) for tid in test_trees  for (f, d) in tree_map[tid]]

    # Stats
    print(f"\n{'='*60}")
    print(f"Output:        {output_dir}")
    print(f"{'='*60}")
    print(f"Unique trees:  {n_trees}")
    print(f"Train trees:   {len(train_trees)}  ({len(train_trees)/n_trees:.1%})  IDs: {sorted(train_trees)}")
    print(f"Test  trees:   {len(test_trees)}   ({len(test_trees)/n_trees:.1%})  IDs: {sorted(test_trees)}")
    print(f"Train folders: {len(train_entries)}")
    print(f"Test  folders: {len(test_entries)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("Train folders:")
        for folder, src in sorted(train_entries):
            print(f"  [{os.path.basename(src)}] {folder}")
        print("\nTest folders:")
        for folder, src in sorted(test_entries):
            print(f"  [{os.path.basename(src)}] {folder}")
        print("\nDRY RUN — no files were copied.")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    def copy_folder(folder_name, source_dir, dest_root):
        src  = os.path.join(source_dir, folder_name)
        dest = os.path.join(dest_root, folder_name)
        if os.path.exists(dest):
            print(f"  [SKIP] already exists: {folder_name}")
            return
        shutil.copytree(src, dest)
        print(f"  [COPY] {folder_name} -> {os.path.basename(dest_root)}/")

    print("Copying train folders...")
    for folder, src in sorted(train_entries):
        copy_folder(folder, src, train_dir)

    print("\nCopying test folders...")
    for folder, src in sorted(test_entries):
        copy_folder(folder, src, test_dir)

    print(f"\nDone. Output written to: {output_dir}")


def main():
    print(f"Mode: {MODE.upper()}")

    if MODE == "partial":
        folders  = collect_folders(PARTIAL_DIR)
        tree_map = build_tree_map(folders, PARTIAL_DIR)

    elif MODE == "merged":
        folders  = collect_folders(MERGED_DIR)
        tree_map = build_tree_map(folders, MERGED_DIR)

    elif MODE == "combined":
        partial_folders = collect_folders(PARTIAL_DIR)
        merged_folders  = collect_folders(MERGED_DIR)
        partial_map     = build_tree_map(partial_folders, PARTIAL_DIR)
        merged_map      = build_tree_map(merged_folders,  MERGED_DIR)
        tree_map        = merge_tree_maps(partial_map, merged_map)

    else:
        raise ValueError(f"Unknown MODE: '{MODE}'. Choose 'partial', 'merged', or 'combined'.")

    perform_split(tree_map, OUTPUT_DIR, TRAIN_RATIO, SEED, DRY_RUN)


if __name__ == "__main__":
    main()