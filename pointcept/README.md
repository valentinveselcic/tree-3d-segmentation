# Pointcept Models: PTv3, PTv2, OctFormer, SparseUNet

This submodule handles training and evaluation of voxelized and serialization-based architectures (Point Transformer v3, Point Transformer v2, OctFormer, SparseUNet, and Stratified Transformer) on the BRANCH v2 dataset.

---

## Class Weights & Loss Parameters

Inverse frequency weights for Weighted Cross-Entropy (1/freq) and Focal Loss parameters (1 - freq) calculated across dataset splits:

### Combined Split
* **CE Weights:** `[1.1704899, 0.17648609, 1.65302401]`
* **Focal Loss Weights:** `[0.8801, 0.2048, 0.9151]`

### Partial Split
* **CE Weights:** `[1.37226482, 0.24165204, 1.38608314]`
* **Focal Loss Weights:** `[0.8696, 0.2595, 0.8709]`

### Merged Split
* **CE Weights:** `[1.0299263, 0.14472738, 1.82534631]`
* **Focal Loss Weights:** `[0.8848, 0.1802, 0.9350]`

---

## Training Commands

Example training commands for the `pointcept` workspace:

### 1. Training from Scratch (Combined Dataset)

```bash

# Point Transformer v3 (Cross-Entropy)
sh scripts/train.sh -p python -d custom_trees -c semseg-pt-v3m1-0-base-no-lovasz -n PTv3-scratch-combined-ce

# Point Transformer v3 (Weighted Cross-Entropy)
sh scripts/train.sh -p python -d custom_trees -c semseg-pt-v3m1-0-base-weighted-CE -n PTv3-scratch-combined-weighted-ce

# OctFormer
sh scripts/train.sh -p python -d custom_trees -c semseg-octformer-v1m1-0-base -n Octformer-scratch-combined-CE

# SparseUNet
sh scripts/train.sh -p python -d custom_trees -c semseg-spunet-v1m1-0-base -n SpUNet-scratch-combined
```

### 2. Dataset Ablations (Partial vs. Merged)

```bash
# PTv3 on Partial Point Clouds
sh scripts/train.sh -p python -d custom_trees -c semseg-pt-v3m1-0-base -n PTv3-scratch-partial

# PTv2 on Partial Point Clouds
sh scripts/train.sh -p python -d custom_trees -c semseg-pt-v2m2-0-base -n PTv2-scratch-partial
```

### 3. Pretrained Backbone Experiments

```bash
# PTv3 with ScanNet200 Pretrained Backbone (3-channel)
sh scripts/train.sh -p python -d custom_trees -c scannet200-semseg-pt-v3m1-0-base -n PTv3-Scannet200-Backbone-3ch

# PTv3 with ScanNet Pretrained Backbone (3-channel)
sh scripts/train.sh -p python -d custom_trees -c scannet-semseg-pt-v3m1-0-base -n PTv3-Scannet-Backbone-3ch-teaser-dataset
```

## Custom Tools

Located in `custom_tools/`:
* `calculate_ratios.py` / `calculate_weights.py` — Recomputes class distribution statistics.
* `fix_stem.py` / `strip_head.py` — Pretrained backbone weight adaptation scripts.
* `split_dataset.py` — Generates dataset partitions.
* `visualize_preds.py` — Generates qualitative Open3D renders for prediction point clouds.