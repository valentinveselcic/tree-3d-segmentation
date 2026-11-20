# Point-Based Models: PointMLP & PointNet++

This submodule contains the training pipelines and evaluation routines for coordinate-based point architectures (PointMLP and PointNet++). These models operate on fixed-size coordinate tensors  using a sliding-window strategy without spatial voxelization.

---

## Training

Example training commands for the `pointmlp` workspace:

### 1. Training from Scratch (Combined Dataset)

```bash
# PointNet++ baseline
python main.py --model pointnet2 \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised \
            --epochs 400 \
            --batch_size 8 \
            --num_points 16384 \
            --wandb_project pointcept

# PointMLP baseline
python main.py --model pointMLP \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised \
            --epochs 400 \
            --batch_size 8 \
            --num_points 16384 \
            --wandb_project pointcept
```

### 2. Loss Function Ablations

```bash
# PointNet++ with Focal Loss
python main.py --model pointnet2 \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised-focal \
            --epochs 400 --batch_size 8 --num_points 16384 \
            --loss focal --focal_gamma 2.0 --focal_alpha 0.892 0.3778 0.891 0.8392 \ ;

# PointMLP with Weighted Cross-Entropy
python main.py --model pointMLP \
            --data_root ../data/partial_revised \
            --exp_name scratch-partial-revised-wce \
            --epochs 400 --batch_size 8 --num_points 16384 \
            --loss weighted_ce --class_weights 1.4104 0.2448 1.3975 0.9473 ;
```

---

## Evaluation & Prediction Export

Example evaluation commands for the `pointmlp` workspace:
```bash
# Evaluate checkpoint performance
python main.py --eval \
               --model pointnet2 \
               --data_root data/partial_revised \
               --exp_name scratch-partial-revised

# Export predictions (.npy / .ply) for qualitative visual analysis
python main.py --save_preds \
               --model pointMLP \
               --data_root data/partial_revised \
               --exp_name scratch-partial-revised \
               --num_points 16384
```
