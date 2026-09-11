# Point-Based Models: PointMLP & PointNet++

This submodule contains the training pipelines and evaluation routines for coordinate-based point architectures (PointMLP and PointNet++). These models operate on fixed-size coordinate tensors  using a sliding-window strategy without spatial voxelization.

---

## Training

Example training commands for the `pointmlp` workspace:

### 1. Training from Scratch (Combined Dataset)

```bash
# PointNet++ baseline
python main.py --model pointnet2 \
               --data_root data/combined \
               --exp_name scratch-combined \
               --epochs 800 \
               --batch_size 8 \
               --num_points 8192 \
               --wandb_project pointcept \
               --resume

# PointMLP baseline
python main.py --model pointMLP \
               --data_root data/combined \
               --exp_name scratch-combined \
               --epochs 800 \
               --batch_size 8 \
               --num_points 8192 \
               --wandb_project pointcept
```

### 2. Loss Function Ablations

```bash
# PointNet++ with Focal Loss
python main.py --model pointnet2 \
               --data_root data/combined \
               --exp_name scratch-combined-focal \
               --epochs 800 \
               --batch_size 8 \
               --num_points 8192 \
               --loss focal \
               --focal_gamma 2.0 \
               --focal_alpha 0.8801 0.2048 0.9151

# PointMLP with Weighted Cross-Entropy
python main.py --model pointMLP \
               --data_root data/combined \
               --exp_name scratch-combined-wce \
               --epochs 800 \
               --batch_size 8 \
               --num_points 8192 \
               --loss weighted_ce \
               --class_weights 1.1704899 0.17648609 1.65302401
```

---

## Evaluation & Prediction Export

Example evaluation commands for the `pointmlp` workspace:
```bash
# Evaluate checkpoint performance
python main.py --eval \
               --model pointnet2 \
               --data_root data/combined \
               --exp_name scratch-combined

# Export predictions (.npy / .ply) for qualitative visual analysis
python main.py --save_preds \
               --model pointMLP \
               --data_root data/combined \
               --exp_name scratch-combined \
               --num_points 8192
```