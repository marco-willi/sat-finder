"""Configuration constants shared across modules.

Single source of truth for all dimensions, paths, and display settings.
These values are also injected into the JavaScript viewer via data attributes.
"""

import os
from pathlib import Path

# =============================================================================
# City Configurations
# =============================================================================

CITIES = {
    "graz": {
        "name": "Graz",
        "img_w": 6656,
        "img_h": 6912,
        "grid_w": 416,
        "grid_h": 432,
        "dzi_url": "/static/tiles_graz/scene.dzi",
        "embeddings_file": "graz_embeddings.npz",
        "map_file": "graz.jpg",
    },
}

DEFAULT_CITY = "graz"

# =============================================================================
# Legacy Image and Embedding Dimensions (for backward compatibility)
# =============================================================================

# Map image dimensions in pixels (Vienna default)
IMG_W = CITIES[DEFAULT_CITY]["img_w"]
IMG_H = CITIES[DEFAULT_CITY]["img_h"]

# DINOv3 embedding grid dimensions (tokens)
GRID_W = CITIES[DEFAULT_CITY]["grid_w"]
GRID_H = CITIES[DEFAULT_CITY]["grid_h"]

# DINO patch size (pixels per token)
PATCH_SIZE = 16

# =============================================================================
# Paths
# =============================================================================

# Base directories - use environment variable or fall back to /app for Docker
APP_ROOT = Path(os.environ.get("APP_ROOT", "/app"))

# Check if we're in development (repo structure) or production (Docker)
PACKAGE_DIR = Path(__file__).parent
_dev_root = PACKAGE_DIR.parent.parent  # src/satfinder -> src -> repo_root

# Use dev root if assets exist there, otherwise use APP_ROOT
if (_dev_root / "assets").exists():
    REPO_ROOT = _dev_root
else:
    REPO_ROOT = APP_ROOT

ASSETS_DIR = REPO_ROOT / "assets"
STATIC_DIR = REPO_ROOT / "static"

# =============================================================================
# URLs (for client-side resources)
# =============================================================================

DZI_URL = "/static/tiles/scene.dzi"
OSD_JS_URL = "/static/js/openseadragon.min.js"

# =============================================================================
# Display Settings
# =============================================================================

# Grid overlay
MIN_GRID_SPACING_PX = 12  # Minimum screen pixels between grid lines
GRID_OPACITY = 0.25

# Heatmap overlay
DEFAULT_HEATMAP_OPACITY = 0.6

# Viewer dimensions
VIEWER_HEIGHT_PX = 700

# =============================================================================
# Algorithm Settings
# =============================================================================

# Weight for negative examples in similarity computation
NEGATIVE_WEIGHT = 0.5
