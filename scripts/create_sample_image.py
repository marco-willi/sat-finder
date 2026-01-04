"""
Create a synthetic sample satellite image for testing.

This generates a simple test image with building-like structures
on a greenish background, simulating a satellite view.
"""

import numpy as np
from PIL import Image
from pathlib import Path


def create_sample_image(size: int = 512, num_buildings: int = 15) -> np.ndarray:
    """
    Create a synthetic satellite-like image with buildings.

    Args:
        size: Image size (square)
        num_buildings: Number of building-like structures

    Returns:
        RGB image as numpy array
    """
    np.random.seed(42)

    # Create base terrain (greenish/brownish)
    image = np.zeros((size, size, 3), dtype=np.uint8)

    # Base color with noise
    base_green = np.random.randint(60, 100, (size, size), dtype=np.uint8)
    base_red = np.random.randint(50, 80, (size, size), dtype=np.uint8)
    base_blue = np.random.randint(40, 70, (size, size), dtype=np.uint8)

    image[:, :, 0] = base_red
    image[:, :, 1] = base_green
    image[:, :, 2] = base_blue

    # Add some road-like features
    for _ in range(3):
        if np.random.random() > 0.5:
            # Horizontal road
            y = np.random.randint(50, size - 50)
            width = np.random.randint(8, 15)
            image[y : y + width, :, :] = [120, 115, 110]
        else:
            # Vertical road
            x = np.random.randint(50, size - 50)
            width = np.random.randint(8, 15)
            image[:, x : x + width, :] = [120, 115, 110]

    # Add buildings (rectangular structures with varied colors)
    building_colors = [
        [180, 175, 170],  # Light gray
        [150, 145, 140],  # Medium gray
        [200, 195, 190],  # White-ish
        [140, 120, 100],  # Brown
        [160, 155, 150],  # Gray
    ]

    buildings = []
    for _ in range(num_buildings):
        # Random building size
        w = np.random.randint(20, 50)
        h = np.random.randint(20, 50)

        # Random position
        x = np.random.randint(10, size - w - 10)
        y = np.random.randint(10, size - h - 10)

        # Check for overlap with existing buildings
        overlap = False
        for bx, by, bw, bh in buildings:
            if x < bx + bw and x + w > bx and y < by + bh and y + h > by:
                overlap = True
                break

        if not overlap:
            buildings.append((x, y, w, h))
            color = building_colors[np.random.randint(0, len(building_colors))]
            # Add slight color variation
            color = [c + np.random.randint(-10, 10) for c in color]
            color = [max(0, min(255, c)) for c in color]

            image[y : y + h, x : x + w, :] = color

            # Add shadow effect
            shadow_offset = 3
            if y + h + shadow_offset < size and x + w + shadow_offset < size:
                shadow_region = image[
                    y + h : y + h + shadow_offset,
                    x + shadow_offset : x + w + shadow_offset,
                    :,
                ]
                image[
                    y + h : y + h + shadow_offset,
                    x + shadow_offset : x + w + shadow_offset,
                    :,
                ] = (shadow_region * 0.7).astype(np.uint8)

    # Add some vegetation patches (darker green)
    for _ in range(20):
        cx = np.random.randint(20, size - 20)
        cy = np.random.randint(20, size - 20)
        radius = np.random.randint(10, 30)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        # Check if not building
                        is_building = False
                        for bx, by, bw, bh in buildings:
                            if bx <= nx < bx + bw and by <= ny < by + bh:
                                is_building = True
                                break
                        if not is_building:
                            image[ny, nx, 1] = min(255, image[ny, nx, 1] + 30)

    return image


def main():
    """Generate and save sample image."""
    output_path = Path(__file__).parent.parent / "app" / "assets" / "sample_tile.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Creating sample image...")
    image = create_sample_image()

    print(f"Saving to {output_path}")
    Image.fromarray(image).save(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
