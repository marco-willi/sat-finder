FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget ca-certificates git git-lfs build-essential pkg-config \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    make graphviz \
    bash bash-completion \
    openssh-client \
    libvips-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Expose ports
EXPOSE 8887 6007 7860

# Default command
CMD ["/bin/bash"]
