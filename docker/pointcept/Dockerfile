FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

# Base packages
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip python3.10-distutils \
    git wget curl ninja-build build-essential \
    libopenblas-dev libsparsehash-dev \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# PyTorch 2.7 with CUDA 12.8
RUN pip install --upgrade pip && \
    pip install torch==2.7.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Core dependencies (matches official Pointcept env)
RUN pip install \
    h5py pyyaml \
    sharedarray tensorboard tensorboardx \
    wandb \
    yapf addict einops scipy plyfile \
    termcolor timm \
    ftfy regex tqdm \
    open3d \
    torch-geometric

# PyG extensions
RUN pip install \
    torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# spconv
RUN pip install spconv-cu126

# Transformers + PEFT (used by newer Pointcept models)
RUN pip install transformers==4.50.3 peft

# CLIP
RUN pip install git+https://github.com/openai/CLIP.git

# Flash Attention
RUN pip install packaging ninja && \
    pip install flash-attn --no-build-isolation

# Build pointops
COPY libs/pointops /tmp/pointops
RUN cd /tmp/pointops && python setup.py install && rm -rf /tmp/pointops

WORKDIR /workspace/Pointcept

CMD ["/bin/bash"]
