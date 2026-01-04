"""FastAPI application factory with static file serving and health endpoint."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse

from .config import STATIC_DIR, ASSETS_DIR, REPO_ROOT


def create_fastapi_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Mounts static file directories and includes API routes.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="SatFinder",
        description="Satellite structure detection using DINOv3 embeddings",
    )

    # Mount static directory for tiles if it exists
    # Note: Do NOT mount /assets - that conflicts with Gradio's frontend assets
    # The assets/ directory is only used internally to load embeddings from disk
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Health check endpoint
    @app.get("/api/health")
    def health():
        """Health check endpoint for monitoring."""
        return {"status": "ok"}

    # Debug endpoint to check paths
    @app.get("/api/debug", response_class=PlainTextResponse)
    def debug():
        """Debug endpoint to check file paths."""
        import os

        lines = [
            f"REPO_ROOT: {REPO_ROOT}",
            f"ASSETS_DIR: {ASSETS_DIR}",
            f"STATIC_DIR: {STATIC_DIR}",
            f"ASSETS_DIR exists: {ASSETS_DIR.exists()}",
            f"STATIC_DIR exists: {STATIC_DIR.exists()}",
            f"CWD: {os.getcwd()}",
        ]
        if ASSETS_DIR.exists():
            lines.append(f"Assets contents: {list(ASSETS_DIR.iterdir())}")
        if STATIC_DIR.exists():
            lines.append(f"Static contents: {list(STATIC_DIR.iterdir())}")
        return "\n".join(lines)

    return app
