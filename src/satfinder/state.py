"""State management for the Gradio application.

Provides event handlers that work with Gradio's state system.
"""

import json

from .config import CITIES
from .similarity import compute_similarity, pixel_to_token
from .visualization import similarity_to_heatmap_base64


def parse_points(points_json: str) -> list[dict]:
    """Parse points JSON string to list of point dicts."""
    if not points_json:
        return []
    try:
        return json.loads(points_json)
    except json.JSONDecodeError:
        return []


def add_point(
    click_xy: str, points_json: str, point_type: str, city: str = "vienna"
) -> tuple[str, str]:
    """Add a point from click coordinates.

    Args:
        click_xy: "x,y" string from JavaScript
        points_json: Current points as JSON string
        point_type: "pos" for positive, "neg" for negative
        city: City name for dimension lookup

    Returns:
        Tuple of (updated_points_json, log_message)
    """
    if not click_xy or click_xy.strip() == "":
        return points_json, ""

    try:
        parts = click_xy.strip().split(",")
        x, y = float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return points_json, f"Invalid click: {click_xy}"

    # Get city dimensions
    city_config = CITIES.get(city, CITIES["vienna"])
    img_w = city_config["img_w"]
    img_h = city_config["img_h"]

    # Validate coordinates
    if not (0 <= x < img_w and 0 <= y < img_h):
        return points_json, f"Click outside image bounds: ({x:.0f}, {y:.0f})"

    # Parse existing points
    points = parse_points(points_json)

    # Add new point
    label = "neg" if point_type == "neg" else "pos"
    points.append({"x": x, "y": y, "label": label})

    tx, ty = pixel_to_token(x, y, city)
    label_text = "negative" if label == "neg" else "positive"
    log = f"Added {label_text} point at pixel ({x:.0f}, {y:.0f}) -> token ({tx}, {ty})"

    return json.dumps(points), log


def remove_last_point(points_json: str) -> tuple[str, str]:
    """Remove the last added point.

    Args:
        points_json: Current points as JSON string

    Returns:
        Tuple of (updated_points_json, log_message)
    """
    points = parse_points(points_json)

    if not points:
        return "[]", "No points to remove"

    removed = points.pop()
    label = "negative" if removed.get("label") == "neg" else "positive"
    log = f"Removed {label} point at ({removed['x']:.0f}, {removed['y']:.0f})"

    return json.dumps(points), log


def clear_points() -> tuple[str, str, str]:
    """Clear all points and heatmap.

    Returns:
        Tuple of (points_json, heatmap_data, log_message)
    """
    return "[]", "", "Points cleared"


def compute_and_return_heatmap(
    points_json: str, city: str = "vienna"
) -> tuple[str, str]:
    """Compute similarity and return heatmap as base64.

    Args:
        points_json: Points as JSON string
        city: City name for embeddings lookup

    Returns:
        Tuple of (heatmap_base64, log_message)
    """
    points = parse_points(points_json)

    if not points:
        return "", "No points selected"

    pos_count = sum(1 for p in points if p.get("label", "pos") == "pos")
    neg_count = sum(1 for p in points if p.get("label") == "neg")

    if pos_count == 0:
        return "", "Need at least one positive point"

    similarity = compute_similarity(points, city)
    heatmap_b64 = similarity_to_heatmap_base64(similarity)

    city_name = CITIES.get(city, {}).get("name", city)
    log = f"[{city_name}] Computed similarity from {pos_count} positive"
    if neg_count > 0:
        log += f" and {neg_count} negative"
    log += " points"

    return heatmap_b64, log


def get_city_config(city: str) -> dict:
    """Get configuration for a city.

    Args:
        city: City key ("vienna" or "graz")

    Returns:
        City configuration dict with img_w, img_h, grid_w, grid_h, dzi_url
    """
    return CITIES.get(city, CITIES["vienna"])
