# Deep Learning-Based Semantic Segmentation of Fruit Tree Point Clouds for Autonomous Agriculture

This repository contains the implementation, training configurations, and experimental evaluation for 3D semantic segmentation of dormant fruit trees using partial single-view RGB-D point clouds.

The project benchmarks six deep learning architectures across transformer, sparse convolutional, and point-based paradigms on a 4-class dataset derived from **BRANCH v2** (663 manually annotated partial point clouds across 58 trees), targeting automated agricultural pruning and orchard infrastructure protection.

---

## Benchmark Results (4-Class Partial Dataset)

All models were trained for 400 epochs using tree-level dataset splits across four semantic categories: **Trunk**, **Branches**, **Support Structures** (trellis wires and concrete poles), and **Grass**.

### Baseline Segmentation (τ = 0 m) & Inference Throughput

| Model | Paradigm | Optimal Loss | mIoU (%) | Trunk IoU (%) | Params (M) | Latency (ms) | Speed (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OctFormer** | Transformer | CE | **95.28** | **87.70** | 44.00 | 16.49 | 60.6 |
| **PTv2** | Transformer | CE | 94.01 | 84.06 | 11.32 | 14.54 | 68.8 |
| **PTv3** | Transformer | CE | 92.85 | 80.59 | 46.17 | 17.15 | 58.3 |
| **SparseUNet** | Sparse CNN | WCE (CE) | 92.50 (92.02) | 80.42 (77.55) | 39.16 | **9.20** | **108.7** |
| **PointMLP** | MLP (Geometric Affine) | CE | 86.65 | 63.67 | 16.75 | 12.26 | 81.6 |
| **PointNet++** | Hierarchical MLP | Focal | 69.52 | 32.80 | **1.74** | 67.49 | 14.8 |

*Note: Single-scene inference latency (median) and throughput (FPS) are measured over 100 forward passes without Test-Time Augmentation (TTA).*

### Boundary-Sensitive Evaluation (τ, θ = 0.02 m, Cross-Entropy)

Because trunk-to-branch transitions lack a sharp physical boundary, models are also evaluated within a 0.02 m tolerance band using **Don't-Care Margin (IoU<sub>DC</sub>)**, **Slack Relaxation (IoU<sub>Sl</sub>)**, and **Boundary F1 (BF<sub>1</sub>)**:

| Model | mIoU<sub>DC</sub> (%) | mIoU<sub>Sl</sub> (%) | Trunk<sub>DC</sub> (%) | Trunk<sub>Sl</sub> (%) | Trunk BF<sub>1</sub> (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OctFormer** | **96.89** | **97.20** | **92.69** | **94.25** | **81.49** |
| **PTv2** | 95.88 | 96.39 | 89.90 | 92.07 | 75.59 |
| **PTv3** | 94.59 | 95.31 | 86.50 | 89.40 | 72.44 |
| **SparseUNet** | 93.90 | 94.80 | 83.39 | 86.99 | 72.08 |

---

## Repository Structure

```text
.
├── docker/
│   ├── pointcept/
│   │   └── Dockerfile          # Environment for PTv3, PTv2, OctFormer, SparseUNet
│   └── pointmlp/
│       └── Dockerfile          # Environment for PointMLP and PointNet++
├── pointcept/                  # Pointcept workspace with custom configs and tools
│   ├── configs/custom_trees/   # 4-class network and loss ablation configs
│   ├── custom_tools/           # Class statistics, boundary metrics, and export scripts
│   ├── data/                   # Tree-level BRANCH v2 partial point cloud splits
│   └── README.md               # Execution commands for Pointcept models
├── pointmlp/                   # PointMLP & PointNet++ workspace
│   ├── data/                   # Tree-level BRANCH v2 partial point cloud splits
│   └── README.md               # Execution commands for MLP-based models
└── README.md                   # Main documentation
```

---

## Hardware & Environment

* **GPU:** NVIDIA GeForce RTX 5070 (12 GB VRAM)
* **CPU:** AMD Ryzen 5 9600X (6 cores / 12 threads)
* **Host OS:** Ubuntu 22.04 LTS
* **Container Runtime:** Docker Engine with NVIDIA Container Toolkit (`nvidia-docker2`)

---

## Pretrained Checkpoints

Trained model weights (`.pth`) are hosted on Hugging Face:  
**[Hugging Face Model Repository](https://huggingface.co/lenlne/tree-3d-segmentation-models)**

---

## Quick Start (Docker Setup)

Due to conflicting CUDA extensions and library dependencies between modern serialization/transformer architectures (`Pointcept`, `SpConv`, `FlashAttention`) and traditional point-based architectures (`PointMLP`, `PointNet++` compiled with custom CUDA ops), the codebase is split into two dedicated Docker environments.

### 1. Build Docker Images

Run all build commands from the repository root:

```bash
# Build Pointcept workspace (PTv3, PTv2, OctFormer, SparseUNet)
docker build -t tree-pointcept:latest -f docker/pointcept/Dockerfile .

# Build PointMLP / PointNet++ workspace
docker build -t tree-pointmlp:latest -f docker/pointmlp/Dockerfile .
```

### 2. Launch Containers

Mount the corresponding workspace directory and allocate sufficient shared memory for PyTorch multi-process dataloading:

```bash
# Launch Pointcept container
docker run --gpus all -it --rm \
  --shm-size=16g \
  -v $(pwd)/pointcept:/workspace/pointcept \
  tree-pointcept:latest bash

# Launch PointMLP / PointNet++ container
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/pointmlp:/workspace/pointmlp \
  tree-pointmlp:latest bash
```

Inside each container, navigate to the target workspace (`cd /workspace/pointcept` or `cd /workspace/pointmlp`) and follow the module documentation:
* [`pointcept/README.md`](pointcept/README.md) — Training, evaluation, boundary metric scripts, and configs for transformer and sparse CNN models.
* [`pointmlp/README.md`](pointmlp/README.md) — Execution routines, loss ablations, and prediction exports for PointMLP and PointNet++.

---

## Acknowledgments

This research was funded by the European Union – NextGenerationEU through the project *"Advanced Machine Learning Methods for Optimization and Robotic Vision"* **SUPORT (581-UNIOS-36)**.

This codebase incorporates and adapts components from:
* [Pointcept](https://github.com/Pointcept/Pointcept)
* [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)
* [OctFormer](https://github.com/octree-nn/octformer)
* [pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch)
* [PointNet++ (erikwijmans)](https://github.com/erikwijmans/Pointnet2_PyTorch)
