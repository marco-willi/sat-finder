"""Visualization utilities for heatmap generation.

This module contains pure visualization logic with no Gradio dependencies.
"""

import base64
import io

import numpy as np
from PIL import Image


def similarity_to_heatmap_base64(similarity: np.ndarray) -> str:
    """Convert similarity map to a base64-encoded PNG image.

    Uses a blue (low) to red (high) colormap via white/purple midpoint.

    Args:
        similarity: (H, W) float array in [0, 1]

    Returns:
        Base64-encoded PNG string
    """
    h, w = similarity.shape

    # Create RGBA image
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Blue to Red colormap via white/purple midpoint
    # Low (0): Blue (0, 0, 255)
    # Mid (0.5): White/Light purple (200, 150, 255)
    # High (1): Red (255, 0, 0)
    t = similarity

    # Red channel: 0 -> 200 -> 255
    rgba[..., 0] = np.where(
        t < 0.5,
        (t * 2 * 200).astype(np.uint8),
        (200 + (t - 0.5) * 2 * 55).astype(np.uint8),
    )

    # Green channel: 0 -> 150 -> 0
    rgba[..., 1] = np.where(
        t < 0.5,
        (t * 2 * 150).astype(np.uint8),
        ((1 - t) * 2 * 150).astype(np.uint8),
    )

    # Blue channel: 255 -> 255 -> 0
    rgba[..., 2] = np.where(
        t < 0.5,
        255,
        ((1 - t) * 2 * 255).astype(np.uint8),
    )

    # Alpha: fully opaque (opacity controlled by CSS/JS)
    rgba[..., 3] = 255

    # Convert to PIL and encode as base64
    img = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def create_colorbar_image(width: int = 20, height: int = 256) -> Image.Image:
    """Create a colorbar image showing the similarity scale.

    Args:
        width: Width of the colorbar in pixels
        height: Height of the colorbar in pixels

    Returns:
        PIL Image of the colorbar
    """
    # Create gradient from 0 to 1
    gradient = np.linspace(1, 0, height).reshape(-1, 1)
    gradient = np.tile(gradient, (1, width))

    # Apply same colormap as heatmap
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    t = gradient

    rgba[..., 0] = np.where(
        t < 0.5,
        (t * 2 * 200).astype(np.uint8),
        (200 + (t - 0.5) * 2 * 55).astype(np.uint8),
    )
    rgba[..., 1] = np.where(
        t < 0.5,
        (t * 2 * 150).astype(np.uint8),
        ((1 - t) * 2 * 150).astype(np.uint8),
    )
    rgba[..., 2] = np.where(
        t < 0.5,
        255,
        ((1 - t) * 2 * 255).astype(np.uint8),
    )
    rgba[..., 3] = 255

    return Image.fromarray(rgba, mode="RGBA")
