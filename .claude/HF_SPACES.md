# Deploying to Hugging Face Spaces

This guide covers how to deploy the sat-finder application to [Hugging Face Spaces](https://huggingface.co/spaces).

## Overview

Hugging Face Spaces provides a simple way to host ML demo apps. For this project, we have three deployment options:

| Option | Best For | Complexity | GPU Support |
|--------|----------|------------|-------------|
| **Gradio** | Quick demos | Low | Yes |
| **Docker** | Full control, production | Medium | Yes |
| **Streamlit** | Current app (requires Docker wrapper) | Medium | Yes |

**Recommended**: Use **Docker** for this project since we have custom dependencies and need GPU support for DINOv3.

## Quick Start

### Option 1: Docker Space (Recommended)

1. Create a new Space at https://huggingface.co/new-space
2. Select **Docker** as SDK
3. Choose hardware (T4 GPU recommended for inference)
4. Push your code with a Dockerfile

### Option 2: Gradio Space

1. Create a new Space with **Gradio** SDK
2. Create `app.py` with Gradio interface
3. Add `requirements.txt`
4. Push to the Space repository

## Docker Space Deployment

### Directory Structure

```
your-space/
├── README.md          # Space configuration (YAML header)
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies
├── app.py            # Application entry point
└── src/              # Source code
    └── satfinder/
```

### README.md Configuration

```yaml
---
title: Satellite Structure Finder
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: t4-small
suggested_storage: small
models:
- timm/vit_large_patch16_dinov3.sat493m
tags:
- satellite-imagery
- similarity-search
- dinov3
pinned: false
---

# Satellite Structure Finder

Find similar structures in satellite imagery using DINOv3 embeddings.
```

### Dockerfile for Streamlit

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user (required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Install the package
RUN pip install --no-cache-dir -e .

# Expose port (must match app_port in README.md)
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/satfinder/streamlit/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
```

### Dockerfile for Gradio

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["python", "app.py"]
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
Pillow>=9.0.0
opencv-python-headless>=4.7.0
streamlit>=1.28.0
streamlit-drawable-canvas-fix>=0.9.3
huggingface_hub>=0.19.0
gradio>=4.0.0  # If using Gradio
```

## Gradio Space Deployment

### Simple Gradio App (app.py)

```python
"""
Gradio app for satellite structure similarity search.
"""
import gradio as gr
import numpy as np
from PIL import Image

from satfinder.similarity import SimilarityEngine
from satfinder.visualization import create_visualization

# Initialize engine (loaded once)
engine = None

def load_engine():
    global engine
    if engine is None:
        engine = SimilarityEngine(backbone="dinov3_sat_large")
    return engine

def find_similar(
    image: Image.Image,
    positive_points: list,
    negative_points: list = None,
    opacity: float = 0.5
):
    """Find similar structures based on clicked points."""
    eng = load_engine()

    # Convert image to numpy
    image_np = np.array(image)

    # Extract features
    eng.extract_features(image_np)

    # Query similarity
    if not positive_points:
        return image_np, "Please select at least one positive point"

    heatmap = eng.query(
        positive_points=positive_points,
        negative_points=negative_points or []
    )

    # Create visualization
    viz = create_visualization(
        image_np,
        heatmap,
        positive_points,
        negative_points or [],
        opacity=opacity
    )

    return viz, f"Found similar regions (max similarity: {heatmap.max():.3f})"

# Create Gradio interface
with gr.Blocks(title="Satellite Structure Finder") as demo:
    gr.Markdown("# 🛰️ Satellite Structure Finder")
    gr.Markdown("Upload a satellite image and click to find similar structures.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Satellite Image",
                type="pil",
                tool="select"  # Enable point selection
            )
            opacity_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                label="Overlay Opacity"
            )
            find_btn = gr.Button("🔍 Find Similar", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Result")
            status = gr.Textbox(label="Status")

    # Example images
    gr.Examples(
        examples=[
            ["examples/vienna_sample.jpg"],
        ],
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

## Hardware Options

### Free Tier
- **CPU Basic**: 2 vCPU, 16 GB RAM, 50 GB disk
- Good for demos with pre-computed embeddings
- Space sleeps after 48 hours of inactivity

### GPU Options (Recommended for this project)

| Hardware | GPU Memory | CPU | RAM | Hourly Price |
|----------|------------|-----|-----|--------------|
| T4 small | 16 GB | 4 vCPU | 15 GB | $0.40 |
| T4 medium | 16 GB | 8 vCPU | 30 GB | $0.60 |
| L4 | 24 GB | 8 vCPU | 30 GB | $0.80 |
| A10G small | 24 GB | 4 vCPU | 14 GB | $1.00 |
| A10G large | 24 GB | 12 vCPU | 46 GB | $1.50 |
| A100 large | 80 GB | 12 vCPU | 142 GB | $2.50 |

**Recommendation**: Start with **T4 small** ($0.40/hr) for this project.

### Persistent Storage

| Tier | Size | Monthly Price |
|------|------|---------------|
| Small | 20 GB | $5 |
| Medium | 150 GB | $25 |
| Large | 1 TB | $100 |

Use persistent storage (`/data` directory) for:
- Pre-computed embeddings
- Cached model weights
- User-uploaded images

## Environment Variables & Secrets

### Setting Variables

In Space Settings > Variables:

```
MODEL_NAME=dinov3_sat_large
EMBEDDINGS_PATH=/data/embeddings.npz
```

### Setting Secrets

For sensitive data (API keys, tokens):

```
HF_TOKEN=hf_xxxxx
```

### Accessing in Code

```python
import os

model_name = os.environ.get("MODEL_NAME", "dinov3_sat_large")
hf_token = os.environ.get("HF_TOKEN")
```

### Docker Secrets (Build Time)

```dockerfile
# Access secret during build
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    huggingface-cli login --token $(cat /run/secrets/HF_TOKEN)
```

## Preloading Models

Use `preload_from_hub` to download models during build:

```yaml
---
title: Satellite Structure Finder
sdk: docker
preload_from_hub:
  - timm/vit_large_patch16_dinov3.sat493m
---
```

## Deployment Steps

### 1. Create the Space

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Create Space
huggingface-cli repo create sat-finder --type space --space-sdk docker
```

### 2. Clone and Configure

```bash
# Clone the Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/sat-finder
cd sat-finder

# Copy project files
cp -r /path/to/satfinder/* .

# Create/update README.md with YAML header
```

### 3. Push to Deploy

```bash
git add .
git commit -m "Initial deployment"
git push
```

The Space will automatically build and deploy.

### 4. Monitor Deployment

- Check build logs in the Space's "Logs" tab
- View runtime logs for debugging
- Monitor hardware usage in Settings

## Optimizing for Spaces

### Pre-compute Embeddings

For faster startup, pre-compute embeddings locally and upload:

```bash
# Pre-compute locally
make precompute-embeddings

# Upload to Space persistent storage or HF Dataset
huggingface-cli upload YOUR_USERNAME/sat-finder-data embeddings.npz
```

### Use Model Caching

```python
import os
from huggingface_hub import hf_hub_download

# Cache models in persistent storage
cache_dir = "/data/models" if os.path.exists("/data") else None

model_path = hf_hub_download(
    repo_id="timm/vit_large_patch16_dinov3.sat493m",
    cache_dir=cache_dir
)
```

### Optimize Docker Image

```dockerfile
# Use slim base image
FROM python:3.11-slim

# Install only needed packages
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Use multi-stage build for smaller image
FROM python:3.11-slim AS runtime
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
```

### Handle Sleep/Wake

Free Spaces sleep after inactivity. Handle gracefully:

```python
import streamlit as st

@st.cache_resource
def load_model():
    """Load model once and cache across sessions."""
    return SimilarityEngine(backbone="dinov3_sat_large")
```

## GPU-Specific Configuration

### PyTorch with CUDA

```
# requirements.txt
--extra-index-url https://download.pytorch.org/whl/cu118
torch
torchvision
```

### Verify GPU Access

```python
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("Running on CPU")
    device = "cpu"
```

## Community GPU Grants

Apply for free GPU access for innovative demos:

1. Go to Space Settings
2. Find "Community GPU Grant" section
3. Submit application describing your project

## Troubleshooting

### Common Issues

**Build fails with permission error:**
```dockerfile
# Ensure user permissions
RUN useradd -m -u 1000 user
USER user
COPY --chown=user . .
```

**Out of memory:**
- Upgrade to larger hardware tier
- Use model quantization
- Load embeddings lazily

**Port not accessible:**
```yaml
# Ensure app_port matches your app
app_port: 7860
```

**Slow startup:**
- Use `preload_from_hub` for models
- Pre-compute embeddings
- Use persistent storage

### Viewing Logs

```bash
# Via CLI
huggingface-cli space logs YOUR_USERNAME/sat-finder

# Or view in web UI under "Logs" tab
```

## Example Spaces

- [Gradio Image Classification](https://huggingface.co/spaces/gradio/image-classification)
- [Docker FastAPI](https://huggingface.co/spaces/DockerTemplates/fastapi-docker)
- [Streamlit Demo](https://huggingface.co/spaces/streamlit/streamlit-example)

## References

- [Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker Spaces Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Gradio Spaces Guide](https://huggingface.co/docs/hub/spaces-sdks-gradio)
- [GPU Upgrades](https://huggingface.co/docs/hub/spaces-gpus)
- [Persistent Storage](https://huggingface.co/docs/hub/spaces-storage)
- [Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
