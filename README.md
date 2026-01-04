---
title: Satellite Structure Finder
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
<!-- The above YAML frontmatter is for Hugging Face Spaces configuration -->

# Satellite Structure Finder 🛰️

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/marco-willi/sat-finder)

**🚀 [Try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/marco-willi/sat-finder)**

Interactive structure detection in satellite imagery using **DINOv3** vision transformers and similarity search. Click on example structures you want to find, and the system will highlight similar regions across the entire map using similarities between DINOv3 embeddings.

![Application Interface](docs/app_ui.jpg)

## Features

- 🎯 **Point-and-click similarity search**: Mark positive/negative examples directly on the map
- 🗺️ **Multi-city support**: Vienna and Graz satellite imagery included
- 🔍 **Deep zoom viewer**: Explore high-resolution satellite imagery with OpenSeadragon
- 🧠 **DINOv3 vision transformer**: State-of-the-art self-supervised feature extraction
- ⚡ **Pre-computed embeddings**: Instant search results (no runtime extraction delay)
- 📊 **Interactive heatmap overlay**: Visualize similarity scores across the map
- 🎨 **DINO grid overlay**: See the 16×16 pixel patch boundaries used by the model
- 📷 **Download functionality**: Export your current view with overlays

## Demo

### Finding Cars in Vienna
<table>
<tr>
<td><img src="docs/vienna_select_cars.png" alt="Select cars" /></td>
<td><img src="docs/vienna_dino_grid.png" alt="DINO grid" /></td>
<td><img src="docs/vienna_find_cars.png" alt="Find similar cars" /></td>
</tr>
<tr>
<td align="center"><b>1. Select Examples</b><br/>Click on cars you want to find</td>
<td align="center"><b>2. View DINO Grid</b><br/>See the 16×16px patches</td>
<td align="center"><b>3. Find Similar</b><br/>Heatmap highlights matches</td>
</tr>
</table>

### Finding Pools in Graz
<table>
<tr>
<td width="50%"><img src="docs/graz_select_pools.png" alt="Select pools" /></td>
<td width="50%"><img src="docs/graz_find_pools.png" alt="Find pools" /></td>
</tr>
<tr>
<td align="center"><b>1. Select Examples</b><br/>Click on swimming pools</td>
<td align="center"><b>2. Find Similar</b><br/>Algorithm finds all pools</td>
</tr>
</table>

## How It Works

### Architecture

```
Satellite Image → DINOv3 Backbone → Dense Embeddings → Similarity Search → Heatmap Overlay
                   (frozen)          (16×16 patches)     (cosine sim)      (Gradio UI)
```

### Algorithm

1. **Pre-computation**: Satellite images are divided into 16×16 pixel patches and encoded using DINOv3 to produce a dense grid of 384-dimensional embeddings
2. **User interaction**: User clicks on structures they want to find (positive examples) and structures to avoid (negative examples)
3. **Query construction**: Selected patch embeddings are averaged to create a query vector (negative examples are subtracted with 0.5 weight)
4. **Similarity search**: Cosine similarity is computed between the query and all patch embeddings
5. **Visualization**: Similarity scores are displayed as a red heatmap overlay on the map

### DINOv3: Self-Supervised Vision Transformer

This project uses [**DINOv3**](https://arxiv.org/abs/2304.07193) (Caron et al., 2023), a state-of-the-art self-supervised vision transformer from Meta AI Research. Specifically, we use the **satellite-adapted variant** `vit_large_patch16_dinov3.sat493m` which was fine-tuned on 493 million satellite/aerial images.

**Key advantages for satellite imagery:**
- **Self-supervised learning**: Trained without labels, learns general visual features
- **Dense features**: Produces embeddings for every 16×16 pixel patch (not just image-level)
- **Satellite adaptation**: Fine-tuned on aerial/satellite imagery for domain-specific features
- **Strong performance**: Excellent at capturing structural patterns (buildings, roads, pools, etc.)

**Model details:**
- Architecture: Vision Transformer (ViT-L/16)
- Input resolution: 224×224 pixels
- Patch size: 16×16 pixels
- Embedding dimension: 1024 → 384 (after projection)
- Parameters: ~304M (backbone)

## Data Sources

### Vienna Orthofoto 2024
- **Source**: Stadt Wien Open Government Data
- **License**: CC BY 4.0
- **Attribution**: "Datenquelle: Stadt Wien – data.wien.gv.at"
- **Resolution**: ~15cm native, ~2.5m/pixel at our zoom level
- **Coverage**: Central Vienna area (5888×7168 pixels)
- **WMTS URL**: https://mapsneu.wien.gv.at/wmtsneu/1.0.0/WMTSCapabilities.xml

### Graz Orthofoto 2024
- **Source**: Stadt Graz Open Government Data
- **License**: CC BY 4.0
- **Attribution**: "Datenquelle: Stadt Graz – data.graz.gv.at"
- **Resolution**: ~15cm native, ~2.5m/pixel at our zoom level
- **Coverage**: Central Graz area (6656×6912 pixels)

Both datasets are publicly available orthophotos (aerial photographs corrected for topographic relief and camera tilt).

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sat-finder.git
cd sat-finder

# Install dependencies
pip install -e ".[dev]"

# Download satellite data and pre-compute embeddings
make download-vienna-stitch    # Download Vienna orthofoto (~16MB)
make download-graz-stitch      # Download Graz orthofoto (~18MB)
make precompute-embeddings     # Generate DINOv3 embeddings (~627MB)
```

### Run the App

```bash
# Start Gradio app (production mode)
make app

# Or use development mode with auto-reload
make app-dev
```

Open your browser to http://localhost:7860

### Usage

1. **Select a city** from the dropdown (Vienna or Graz)
2. Choose **point type**: Positive (+) or Negative (-)
3. **Click on structures** in the map to mark examples
   - Green markers = positive examples (what you want to find)
   - Red markers = negative examples (what to avoid)
4. Click **"Find Similar"** to compute similarity
5. Similar areas will be highlighted in the **red heatmap**
6. Toggle overlays:
   - **Show Heatmap**: Display/hide similarity scores
   - **Show DINO Grid**: Visualize 16×16 patch boundaries
7. Adjust **Heatmap Opacity** for better visualization
8. Use **"Download View"** to save current view as PNG
9. Click **"Clear Points"** to reset and start over

## Development

### Project Structure

```
sat-finder/
├── app.py                      # Main entry point (FastAPI + Gradio)
├── src/satfinder/              # Source code
│   ├── api.py                  # FastAPI application factory
│   ├── config.py               # Configuration constants
│   ├── similarity.py           # Similarity search engine
│   ├── state.py                # Gradio state management
│   └── ui/                     # Gradio UI components
│       ├── controls.py         # Sidebar controls
│       ├── layout.py           # Main Blocks layout
│       └── viewer.py           # OpenSeadragon viewer
├── static/                     # Static assets
│   ├── viewer.html             # OpenSeadragon viewer iframe
│   ├── js/                     # JavaScript libraries
│   ├── tiles/                  # DeepZoom tiles (Vienna)
│   └── tiles_graz/             # DeepZoom tiles (Graz)
├── assets/                     # Pre-computed data
│   ├── vienna_embeddings.npz   # Vienna DINOv3 features
│   ├── graz_embeddings.npz     # Graz DINOv3 features
│   ├── vienna.jpg              # Vienna source image
│   └── graz.jpg                # Graz source image
├── scripts/                    # Data preparation scripts
│   ├── download_vienna_data.py
│   ├── download_graz_data.py
│   └── precompute_embeddings.py
└── docs/                       # Documentation & figures
```

### Key Commands

```bash
make help                  # Show all available commands
make app                   # Start Gradio app (production)
make app-dev               # Start Gradio app (dev mode with auto-reload)
make tiles                 # Generate DeepZoom tiles from images
make precompute-embeddings # Pre-compute DINOv3 embeddings
make tests                 # Run tests
make lint                  # Run linter
make format                # Format code
```

### DevContainer

The project includes a complete VS Code DevContainer setup with GPU support:

```bash
# All dependencies are automatically installed on container creation:
# - Python packages (from pyproject.toml)
# - OpenSeadragon JavaScript library
# - Jupyter kernel setup
# - Pre-commit hooks
```

See [`.devcontainer/post-create.sh`](.devcontainer/post-create.sh) for details.

## Technical Details

### DINOv3 Feature Extraction

```python
from satfinder.similarity import load_embeddings, compute_similarity

# Load pre-computed embeddings (cached)
embeddings = load_embeddings("vienna")  # Shape: (448, 368, 384)
# 448×368 grid of 384-dimensional embeddings

# Compute similarity between query points and all patches
pos_points = [(100, 200), (150, 250)]  # (x, y) coordinates
neg_points = [(300, 400)]

heatmap = compute_similarity(
    embeddings,
    pos_points=pos_points,
    neg_points=neg_points,
    img_w=5888,
    img_h=7168,
    grid_w=368,
    grid_h=448
)
```

### Similarity Scoring

The similarity score for each patch is computed as:

```
query = mean(positive_embeddings) - 0.5 × mean(negative_embeddings)
similarity = cosine_similarity(query, patch_embedding)
```

Where cosine similarity is:
```
cos_sim(A, B) = (A · B) / (||A|| × ||B||)
```

Scores range from -1 (opposite) to +1 (identical). The heatmap visualizes scores above a threshold (typically 0.5).

## Performance

- **Embedding pre-computation**: ~5-10 minutes per city (one-time cost)
- **Search latency**: <100ms for similarity computation (CPU)
- **Memory usage**: ~2GB RAM (embeddings + model)
- **Model size**: ~1.2GB (DINOv3 ViT-L/16)

## Limitations

- Fixed patch size (16×16 pixels) may miss very small or very large structures
- Performance depends on visual similarity (struggles with abstract patterns)
- Pre-computed embeddings are city-specific (requires re-computation for new images)
- Requires significant disk space for embeddings (~627MB per city)

## Citation

If you use this project, please cite:

```bibtex
@misc{simeoni_dinov3_2025,
	title = {{DINOv3}},
	url = {http://arxiv.org/abs/2508.10104},
	doi = {10.48550/arXiv.2508.10104},
	urldate = {2025-09-25},
	publisher = {arXiv},
	author = {Siméoni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Michaël and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timothée and Moutakanni, Théo and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and Jégou, Hervé and Labatut, Patrick and Bojanowski, Piotr},
	month = aug,
	year = {2025},
}
```



For the satellite-adapted DINOv3 variant, see: https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m

## License

MIT License - see [LICENSE](LICENSE) for details.

Data sources (Vienna and Graz orthophotos) are licensed under CC BY 4.0 by their respective cities.

## Acknowledgments

- **DINOv2/DINOv3**: Meta AI Research (Simeoni et al., 2025)
- **Satellite DINOv3**: timm library and Ross Wightman
- **Vienna Orthofoto**: Stadt Wien Open Government Data
- **Graz Orthofoto**: Stadt Graz Open Government Data
- **OpenSeadragon**: Deep zoom viewer for high-resolution imagery
- **Gradio**: Interactive web interface framework

## Contact

For questions or issues, please open an issue on GitHub.
