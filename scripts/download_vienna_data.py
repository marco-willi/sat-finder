"""
Download Vienna Orthofoto tiles from WMTS service.

Uses the City of Vienna's WMTS endpoint to download orthophoto tiles
for a specified area and zoom level.

License: Data from Stadt Wien under CC BY 4.0
Attribution: "Datenquelle: Stadt Wien – data.wien.gv.at"
"""

import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image
from tqdm import tqdm


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates at given zoom level (Web Mercator)."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_latlon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon (upper-left corner)."""
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def get_tile_url(
    x: int, y: int, zoom: int, layer: str = "lb", style: str = "farbe"
) -> str:
    """Generate WMTS tile URL for Vienna Orthofoto."""
    return f"https://mapsneu.wien.gv.at/wmts/{layer}/{style}/google3857/{zoom}/{y}/{x}.jpeg"


def download_tile(
    x: int, y: int, zoom: int, output_dir: Path, layer: str = "lb"
) -> Tuple[int, int, bool]:
    """Download a single tile."""
    url = get_tile_url(x, y, zoom, layer)
    # Organize tiles by zoom level in subdirectories
    zoom_dir = output_dir / f"zoom_{zoom}"
    zoom_dir.mkdir(parents=True, exist_ok=True)
    output_path = zoom_dir / f"tile_x{x}_y{y}.jpg"

    # Skip if already downloaded
    if output_path.exists():
        return x, y, True

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return x, y, True
    except Exception as e:
        print(f"Error downloading tile ({x}, {y}): {e}")
        return x, y, False


def download_tiles_bbox(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int,
    output_dir: Path,
    layer: str = "lb",
    max_workers: int = 10,
) -> None:
    """Download all tiles within a bounding box."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get tile range
    x_min, y_max = latlon_to_tile(min_lat, min_lon, zoom)
    x_max, y_min = latlon_to_tile(max_lat, max_lon, zoom)

    # Calculate total tiles
    tiles_to_download = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles_to_download.append((x, y))

    total_tiles = len(tiles_to_download)
    print(f"Downloading {total_tiles} tiles at zoom level {zoom}")
    print(f"Tile range: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")
    print(f"Output directory: {output_dir}")

    # Download tiles in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_tile, x, y, zoom, output_dir, layer): (x, y)
            for x, y in tiles_to_download
        }

        with tqdm(total=total_tiles, desc="Downloading tiles") as pbar:
            for future in as_completed(futures):
                x, y, success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)

    print(f"\nDownload complete: {successful} successful, {failed} failed")


def stitch_tiles(
    tile_dir: Path,
    output_path: Path,
    zoom: int,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
) -> None:
    """Stitch tiles into a single image."""
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Assume 256x256 tiles (standard for Web Mercator)
    tile_size = 256
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size

    print(f"Stitching tiles into {width}x{height} image...")
    stitched = Image.new("RGB", (width, height))

    # Look in zoom-specific subdirectory
    zoom_dir = tile_dir / f"zoom_{zoom}"
    missing_tiles = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_path = zoom_dir / f"tile_x{x}_y{y}.jpg"
            if tile_path.exists():
                tile = Image.open(tile_path)
                stitched.paste(tile, ((x - x_min) * tile_size, (y - y_min) * tile_size))
            else:
                missing_tiles += 1

    if missing_tiles > 0:
        print(f"Warning: {missing_tiles} tiles were missing during stitching")

    stitched.save(output_path, "JPEG", quality=95)
    print(f"Saved stitched image to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Vienna Orthofoto tiles from WMTS service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download central Vienna at zoom 16 (good for POC)
  python scripts/download_vienna_data.py --preset central --zoom 16

  # Download specific area
  python scripts/download_vienna_data.py --bbox 48.20 48.22 16.36 16.38 --zoom 17

  # Download and stitch into single image
  python scripts/download_vienna_data.py --preset central --zoom 15 --stitch

Zoom levels (approximate resolution):
  14: ~10m/pixel   - Good for city overview
  15: ~5m/pixel    - Neighborhoods
  16: ~2.5m/pixel  - Districts (fast download)
  17: ~1.2m/pixel  - Detailed
  18: ~60cm/pixel  - High detail (recommended for structures)
  19: ~30cm/pixel  - Very high detail
  20: ~15cm/pixel  - Near native resolution (many tiles!)

License: Data from Stadt Wien under CC BY 4.0
Attribution: "Datenquelle: Stadt Wien – data.wien.gv.at"
        """,
    )

    parser.add_argument(
        "--preset",
        choices=["central", "full"],
        help="Use preset bounding box (central Vienna or full city)",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"),
        help="Custom bounding box (lat/lon)",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=18,
        help="Zoom level (14-20 recommended, default: 18)",
    )
    parser.add_argument(
        "--layer",
        default="lb",
        choices=["lb", "lb2023", "lb2022"],
        help="Orthofoto layer (lb=2024, default: lb)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/vienna/orthofoto"),
        help="Output directory for tiles",
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Stitch tiles into single image after download",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel download workers (default: 10)",
    )

    args = parser.parse_args()

    # Define presets
    presets = {
        # Central Vienna: Stephansdom, Hofburg, Ringstrasse area
        "central": (48.195, 48.220, 16.355, 16.385),
        # Full Vienna extent (will download many tiles!)
        "full": (48.10, 48.33, 16.17, 16.58),
    }

    # Get bounding box
    if args.preset:
        min_lat, max_lat, min_lon, max_lon = presets[args.preset]
        print(f"Using preset: {args.preset}")
    elif args.bbox:
        min_lat, max_lat, min_lon, max_lon = args.bbox
    else:
        parser.error("Must specify either --preset or --bbox")

    print(f"Bounding box: ({min_lat}, {min_lon}) to ({max_lat}, {max_lon})")
    print(f"Zoom level: {args.zoom}")
    print(
        f"Layer: {args.layer} (Orthofoto 2024)"
        if args.layer == "lb"
        else f"Layer: {args.layer}"
    )

    # Download tiles
    download_tiles_bbox(
        min_lat,
        max_lat,
        min_lon,
        max_lon,
        args.zoom,
        args.output_dir,
        args.layer,
        args.workers,
    )

    # Optional stitching
    if args.stitch:
        x_min, y_max = latlon_to_tile(min_lat, min_lon, args.zoom)
        x_max, y_min = latlon_to_tile(max_lat, max_lon, args.zoom)
        output_path = args.output_dir / f"vienna_z{args.zoom}_stitched.jpg"
        stitch_tiles(
            args.output_dir, output_path, args.zoom, (x_min, x_max), (y_min, y_max)
        )


if __name__ == "__main__":
    main()
