"""Satellite Structure Finder - Application Entry Point.

Interactive structure detection in satellite imagery using DINOv3 embeddings.

This is the HuggingFace Spaces entrypoint.

Run with:
    python app.py           # Production (uvicorn)
    gradio app.py           # Development (auto-reload)
"""

import gradio as gr

from satfinder.api import create_fastapi_app
from satfinder.config import CITIES, ASSETS_DIR
from satfinder.similarity import load_embeddings
from satfinder.ui import create_gradio_app

# =============================================================================
# Application Setup
# =============================================================================

# Pre-load embeddings for available cities at startup
print(f"Looking for assets in: {ASSETS_DIR}")
print(f"Assets directory exists: {ASSETS_DIR.exists()}")
if ASSETS_DIR.exists():
    print(f"Assets contents: {list(ASSETS_DIR.iterdir())}")

for city, cfg in CITIES.items():
    embeddings_path = ASSETS_DIR / cfg["embeddings_file"]
    if embeddings_path.exists():
        print(f"Loading embeddings for {city}...")
        load_embeddings(city)
    else:
        print(f"Skipping {city} - embeddings not found at {embeddings_path}")

# Create FastAPI app (includes static mounts and /api/health)
fastapi_app = create_fastapi_app()

# Create Gradio app
demo = create_gradio_app()

# Mount Gradio on FastAPI (for static file serving)
app = gr.mount_gradio_app(fastapi_app, demo, path="/")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
