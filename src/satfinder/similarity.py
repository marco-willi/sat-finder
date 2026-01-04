"""Similarity computation using DINOv3 embeddings.

This module contains pure business logic with no Gradio dependencies.
"""

from pathlib import Path

import numpy as np
import yaml
from torchvision import transforms

from .config import (
    ASSETS_DIR,
    NEGATIVE_WEIGHT,
)


def load_model_config() -> dict:
    """Load model configuration from config/models.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_normalization_transform(backbone_name: str) -> transforms.Normalize:
    """
    Get normalization transform for a backbone from config.

    Args:
        backbone_name: Name of the backbone (e.g., 'dinov3_sat_large')

    Returns:
        torchvision Normalize transform with appropriate mean/std
    """
    config = load_model_config()

    # Get backbone config
    backbone_config = config["backbones"].get(backbone_name)
    if backbone_config is None:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Get normalization preset name
    norm_preset = backbone_config.get("normalization", "imagenet")

    # Get normalization values from presets
    norm_config = config["normalization_presets"].get(norm_preset)
    if norm_config is None:
        raise ValueError(f"Unknown normalization preset: {norm_preset}")

    return transforms.Normalize(mean=norm_config["mean"], std=norm_config["std"])


# Cache for embeddings per city
_embeddings_cache: dict[str, np.ndarray] = {}


def load_embeddings(city: str = "vienna") -> np.ndarray:
    """Load pre-computed DINOv3 embeddings for a specific city.

    Args:
        city: City name ("vienna" or "graz")

    Returns:
        Embeddings array of shape (GRID_H, GRID_W, embedding_dim).
        Returns zeros if embeddings file not found.
    """
    from .config import CITIES

    if city in _embeddings_cache:
        return _embeddings_cache[city]

    city_config = CITIES.get(city)
    if not city_config:
        print(f"Warning: unknown city '{city}', using vienna")
        city = "vienna"
        city_config = CITIES["vienna"]

    emb_path = ASSETS_DIR / city_config["embeddings_file"]
    if emb_path.exists():
        data = np.load(emb_path)
        embeddings = data["features"]  # (H', W', D)
        print(f"Loaded {city} embeddings: shape={embeddings.shape}")
        _embeddings_cache[city] = embeddings
        return embeddings
    else:
        print(f"Warning: embeddings not found at {emb_path}")
        grid_h = city_config["grid_h"]
        grid_w = city_config["grid_w"]
        return np.zeros((grid_h, grid_w, 1024), dtype=np.float32)


def pixel_to_token(x: float, y: float, city: str = "vienna") -> tuple[int, int]:
    """Convert pixel coordinates to token grid indices.

    Args:
        x: X coordinate in image pixels
        y: Y coordinate in image pixels
        city: City name for dimension lookup

    Returns:
        Tuple of (token_x, token_y) indices, clamped to valid range.
    """
    from .config import CITIES

    city_config = CITIES.get(city, CITIES["vienna"])
    img_w = city_config["img_w"]
    img_h = city_config["img_h"]
    grid_w = city_config["grid_w"]
    grid_h = city_config["grid_h"]

    token_x = int(x / (img_w / grid_w))
    token_y = int(y / (img_h / grid_h))
    # Clamp to valid range
    token_x = max(0, min(grid_w - 1, token_x))
    token_y = max(0, min(grid_h - 1, token_y))
    return token_x, token_y


def compute_similarity(points: list[dict], city: str = "vienna") -> np.ndarray:
    """Compute similarity map given a list of positive/negative points.

    Uses cosine similarity between query embedding(s) and all patch embeddings.
    Positive points are averaged, negative points are subtracted with weight.

    Args:
        points: List of {"x": float, "y": float, "label": "pos"|"neg"}
        city: City name for embeddings lookup

    Returns:
        Similarity map as (GRID_H, GRID_W) float array in [0, 1]
    """
    from .config import CITIES

    city_config = CITIES.get(city, CITIES["vienna"])
    grid_w = city_config["grid_w"]
    grid_h = city_config["grid_h"]

    embeddings = load_embeddings(city)  # (H', W', D)

    # Separate positive and negative points
    pos_points = [p for p in points if p.get("label", "pos") == "pos"]
    neg_points = [p for p in points if p.get("label") == "neg"]

    if not pos_points:
        return np.zeros((grid_h, grid_w), dtype=np.float32)

    # Get embeddings at positive point locations
    pos_vectors = []
    for p in pos_points:
        tx, ty = pixel_to_token(p["x"], p["y"], city)
        pos_vectors.append(embeddings[ty, tx])

    # Average positive query vector
    query = np.mean(pos_vectors, axis=0)  # (D,)
    query = query / (np.linalg.norm(query) + 1e-8)

    # If we have negative points, subtract their direction
    if neg_points:
        neg_vectors = []
        for p in neg_points:
            tx, ty = pixel_to_token(p["x"], p["y"], city)
            neg_vectors.append(embeddings[ty, tx])
        neg_mean = np.mean(neg_vectors, axis=0)
        neg_mean = neg_mean / (np.linalg.norm(neg_mean) + 1e-8)
        # Subtract negative direction
        query = query - NEGATIVE_WEIGHT * neg_mean
        query = query / (np.linalg.norm(query) + 1e-8)

    # Normalize all embeddings
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
    embeddings_norm = embeddings / norms

    # Cosine similarity: (H', W', D) @ (D,) -> (H', W')
    similarity = np.einsum("hwd,d->hw", embeddings_norm, query)

    # Normalize to [0, 1]
    sim_min, sim_max = similarity.min(), similarity.max()
    if sim_max > sim_min:
        similarity = (similarity - sim_min) / (sim_max - sim_min)
    else:
        similarity = np.zeros_like(similarity)

    return similarity.astype(np.float32)
