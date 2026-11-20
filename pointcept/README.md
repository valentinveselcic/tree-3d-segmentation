# Pointcept Models: PTv3, PTv2, OctFormer, SparseUNet

This submodule handles training and evaluation of voxelized and serialization-based architectures (Point Transformer v3, Point Transformer v2, OctFormer, SparseUNet, and Stratified Transformer) on the BRANCH v2 dataset.

---

## Class Weights & Loss Parameters

Inverse frequency weights for Weighted Cross-Entropy (1/freq) and Focal Loss parameters (1 - freq) calculated across dataset splits:

### Partial Split
* **CE Weights:** `[1.4104, 0.2448, 1.3975, 0.9473]`
* **Focal Loss Weights:** `[0.892, 0.3778, 0.891, 0.8392]`
---

## Training Commands

Example training commands for the `pointcept` workspace:

### Training from Scratch

```bash

# Point Transformer v3 (Cross-Entropy)
sh scripts/train.sh -p python -d custom_trees -c semseg-pt-v3m1-0-base-new -n PTv3-scratch-partial-revised

# Point Transformer v3 (Weighted Cross-Entropy)
sh scripts/train.sh -p python -d custom_trees -c semseg-pt-v3m1-0-base-new-WCE -n PTv3-scratch-partial-revised-WCE

# OctFormer
sh scripts/train.sh -p python -d custom_trees -c semseg-octformer-v1m1-0-base-new -n Octformer-scratch-partial-revised

# SparseUNet
sh scripts/train.sh -p python -d custom_trees -c semseg-spunet-v1m1-0-base-new -n spunet-scratch-partial-revised
```


## Custom Tools

Located in `custom_tools/`:
* `calculate_ratios.py` / `calculate_weights.py` — Recomputes class distribution statistics.
* `fix_stem.py` / `strip_head.py` — Pretrained backbone weight adaptation scripts.
* `split_dataset.py` — Generates dataset partitions.
* `visualize_preds.py` — Generates qualitative Open3D renders for prediction point clouds.
