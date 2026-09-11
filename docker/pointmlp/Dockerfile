FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip python3.10-distutils \
    git wget curl ninja-build build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN pip install --upgrade pip && \
    pip install torch==2.7.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
    cycler einops h5py pyyaml \
    scikit-learn scipy tqdm \
    matplotlib numpy

RUN git clone https://github.com/ma-xu/pointMLP-pytorch.git /workspace/pointMLP

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/pointMLP

COPY patch_setup.py /tmp/patch_setup.py
RUN python3 /tmp/patch_setup.py

RUN pip install --no-build-isolation pointnet2_ops_lib/.

CMD ["/bin/bash"]
