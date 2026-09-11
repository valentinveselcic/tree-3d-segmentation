# 3D Semantic Segmentation of Dormant Fruit Trees Using Point Clouds

This repository contains the implementation, training configurations, and experimental evaluation for 3D semantic segmentation of dormant fruit trees using RGB-D point clouds.

The project benchmarks six deep learning architectures across transformer, sparse convolutional, and point-based paradigms on the **BRANCH v2** dataset, targeting automated agricultural pruning and infrastructure protection.

---

## Benchmark Results (Combined Dataset)

All models were evaluated through an 800-epoch training schedule under identical test protocols on three target classes: **Trunk (D)**, **Branch (G)**, and **Support Structures (PS)**.

| Model | Paradigm | mIoU (%) | IoU-D (%) | IoU-G (%) | IoU-PS (%) | mAcc (%) | allAcc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OctFormer** | Transformer | **93.37** | **86.23** | **97.38** | 96.51 | **96.07** | **97.88** |
| **PTv2** | Transformer | 93.17 | 85.19 | 97.27 | 97.06 | 95.97 | 97.78 |
| **PTv3** | Transformer | 93.11 | 84.82 | 97.29 | **97.21** | 95.66 | 97.78 |
| **SparseUNet** | Sparse CNN | 92.79 | 84.14 | 97.12 | 97.12 | 95.57 | 97.65 |
| **PointMLP** | MLP (Geometric Affine) | 73.63 | 48.34 | 93.48 | 79.08 | 81.69 | 93.85 |
| **PointNet++** | Hierarchical MLP | 48.54 | 18.19 | 89.03 | 38.41 | 54.21 | 88.93 |

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
│   ├── configs/custom_trees/   # Network configuration files
│   ├── custom_tools/           # Class statistics and qualitative export scripts
│   ├── data/                   # Embedded BRANCH v2 dataset splits
│   └── README.md               # Execution commands for Pointcept models
├── pointmlp/                   # PointMLP & PointNet++ workspace
│   ├── data/                   # Embedded BRANCH v2 dataset splits
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

All trained model weights (`.pth`) are hosted on Hugging Face:  
🔗 **[Hugging Face Model Repository](https://huggingface.co/lenlne/tree-3d-segmentation-models)**

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
* [`pointcept/README.md`](pointcept/README.md) — Training, evaluation, and configuration scripts for transformer and sparse CNN models.
* [`pointmlp/README.md`](pointmlp/README.md) — Execution routines, loss ablations, and prediction exports for PointMLP and PointNet++.

---

## Acknowledgments

This research incorporates and adapts components from:
* [Pointcept](https://github.com/Pointcept/Pointcept)
* [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)
* [OctFormer](https://github.com/octree-nn/octformer)
* [pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch)
* [PointNet++ (erikwijmans)](https://github.com/erikwijmans/Pointnet2_PyTorch)