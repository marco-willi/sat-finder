# sat-finder

Interactive structure detection in satellite imagery using DINOv3 embeddings and similarity search.

## Architecture

```

**Note**: OpenSeadragon is automatically downloaded during devcontainer setup (see `.devcontainer/post-create.sh`).
Image → DINOv3 Backbone (frozen) → Dense Features → Similarity Engine → Gradio Web UI
```

**Note**: OpenSeadragon is automatically downloaded during devcontainer setup (see `.devcontainer/post-create.sh`).

User clicks on example structures (positive/negative), system finds similar regions via cosine similarity on patch embeddings.

## Quick Start

```bash
pip install -e ".[dev]"
make download-vienna-stitch    # Download Vienna orthofoto sample
make precompute-embeddings     # Pre-compute DINOv3 embeddings
make app                       # Launch Gradio app at http://localhost:7860
```

**Note**: OpenSeadragon is automatically downloaded during devcontainer setup (see `.devcontainer/post-create.sh`).

## Data

**Vienna Orthofoto 2024** from Stadt Wien (CC BY 4.0)
- Attribution: "Datenquelle: Stadt Wien – data.wien.gv.at"
- Downloaded via WMTS: https://mapsneu.wien.gv.at/wmtsneu/1.0.0/WMTSCapabilities.xml
- Resolution: ~15cm native, ~2.5m/pixel at zoom 16

**Graz Orthofoto 2024** from Stadt Graz (CC BY 4.0)
- Available for comparison and testing

```bash
make download-vienna         # Download central Vienna tiles
make download-vienna-stitch  # Download and stitch into single image
make download-graz           # Download central Graz tiles
make download-graz-stitch    # Download and stitch into single image
```

**Note**: OpenSeadragon is automatically downloaded during devcontainer setup (see `.devcontainer/post-create.sh`).

## First Steps (POC Checklist)

- [x] Implement `src/satfinder/features.py` - DINOv3 feature extraction
- [x] Implement `src/satfinder/similarity.py` - Similarity search engine
- [x] Implement `src/satfinder/visualization.py` - Heatmap overlay utilities
- [x] Create Gradio application with modular structure
- [x] Pre-compute embeddings for instant similarity search
- [x] Test end-to-end flow

## Project Structure

```

**Note**: OpenSeadragon is automatically downloaded during devcontainer setup (see `.devcontainer/post-create.sh`).
src/satfinder/
├── __init__.py           # Package exports
├── api.py                # FastAPI application factory
├── config.py             # Configuration constants
├── similarity.py         # Similarity search engine
├── state.py              # Gradio state management
└── ui/                   # Gradio UI components
    ├── controls.py       # Sidebar controls
    ├── layout.py         # Main Blocks layout
    └── viewer.py         # OpenSeadragon viewer

app.py                    # Main entry point (mounts Gradio on FastAPI)
static/                   # Static assets
├── js/                   # JavaScript libraries (OpenSeadragon)
├── tiles/                # DeepZoom tiles for Vienna
├── tiles_graz/           # DeepZoom tiles for Graz
└── viewer.html           # OpenSeadragon viewer iframe

assets/                   # Pre-computed embeddings and images
├── vienna_embeddings.npz # Pre-computed DINOv3 features for Vienna
├── graz_embeddings.npz   # Pre-computed DINOv3 features for Graz
├── vienna.jpg            # Vienna orthofoto source image
└── graz.jpg              # Graz orthofoto source image

archive/
└── src_archive/satfinder/streamlit/  # Archived Streamlit app
```

**Note**: OpenSeadragon is automatically downloaded during devcontainer setup (see `.devcontainer/post-create.sh`).

## Key Implementation Details

- **Model**: `vit_large_patch16_dinov3.sat493m` (via timm)
- **Patch size**: 16 → features at (H/16, W/16) resolution
- **Similarity**: Cosine similarity between query embedding and all patch embeddings
- **Negative examples**: Subtracted from query with 0.5 weight
- **Viewer**: OpenSeadragon for deep zoom tile navigation
- **Multi-city support**: Vienna and Graz datasets with independent embeddings

## Key Commands

```bash
make help                  # Show available commands
make app                   # Run Gradio app (production mode with uvicorn)
make app-dev               # Run Gradio app (development mode with auto-reload)
make tiles                 # Generate DeepZoom tiles from map images
pytest                     # Run tests
```

**Note**: OpenSeadragon is automatically downloaded during devcontainer setup (see `.devcontainer/post-create.sh`).

## Deployment (Dual Remote Setup)

The project uses two git remotes with different content:

| Remote | URL | Content |
|--------|-----|---------|
| `origin` | github.com/marco-willi/sat-finder | Full repo (Vienna + Graz) |
| `hf` | huggingface.co/spaces/marco-willi/sat-finder | Graz only (1GB limit) |

### Why Dual Remotes?

HF Spaces free tier has a **1GB storage limit**. Vienna data (~627MB embeddings + tiles) exceeds this when combined with Graz (~684MB). So HF only gets Graz data.

### Push Workflow

**For normal changes (code only):**
```bash
git add -A && git commit -m "message" && git push origin main
# Then cherry-pick to HF:
git checkout -b hf-update hf/main
git cherry-pick main --strategy-option theirs
git push hf hf-update:main --force
git checkout main && git branch -D hf-update
```

**Why cherry-pick?** The two remotes have diverged histories (different initial commits). Direct push fails because GitHub's main includes Vienna files that HF can't accept.

### Files Excluded from HF (via .hfignore)
- `archive/` - Old code
- `notebooks/` - Development notebooks
- `scripts/` - Data download scripts
- `docs/` - Documentation images
- `config/` - Model configs
- `data/` - Raw data cache
- `logs/` - Log files
- `.devcontainer/` - Dev environment

### Adding Vienna to HF (requires paid tier)
If you upgrade HF storage, add Vienna back to `CITIES` in `config.py` and push the Vienna files.


## Gradio App Features

The interactive Gradio application provides:
- **Multi-city support**: Switch between Vienna and Graz satellite imagery
- **Point-based similarity search**: Click to mark positive (+) and negative (-) example structures
- **Deep zoom viewer**: OpenSeadragon-powered tile navigation for high-resolution imagery
- **Download functionality**: Capture and download current map view with overlays
- **Similarity heatmap visualization**: Real-time heatmap overlay with adjustable opacity
- **Patch grid overlay**: Visualize DINOv3's 16×16 pixel patch boundaries
- **Pre-computed embeddings**: Instant similarity search using cached features (no extraction delay)

### Usage
1. **Select a city** from the dropdown (Vienna or Graz)
2. Choose point type: "Positive (+)" or "Negative (-)"
3. **Click on structures** in the map to mark examples
4. Click **"Find Similar"** to compute similarity across entire map
5. Similar areas will be highlighted in red heatmap
6. Toggle "Show Heatmap" and "Show DINO Grid" overlays
7. Adjust "Heatmap Opacity" slider for better visualization
8. Use **"Download View"** button to save current view as PNG
9. Click **"Clear Points"** to reset and start over

## References

See [SIMILARITY_INTERACTIVE.md](SIMILARITY_INTERACTIVE.md) for detailed implementation plan.
