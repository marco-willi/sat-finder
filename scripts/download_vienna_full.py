#!/usr/bin/env python3
"""
Download full Vienna Orthofoto tiles from WMTS service.

Downloads the complete Vienna city extent at specified zoom level.
Saves to a separate directory to avoid overwriting existing data.

Estimated tile counts at different zoom levels:
  - Zoom 16: ~4,500 tiles (~280 MB)
  - Zoom 17: ~18,000 tiles (~1.1 GB)
  - Zoom 18: ~72,000 tiles (~4.5 GB)
  - Zoom 19: ~288,000 tiles (~18 GB)

License: Data from Stadt Wien under CC BY 4.0
Attribution: "Datenquelle: Stadt Wien – data.wien.gv.at"

Usage:
    python scripts/download_vienna_full.py --zoom 17
    python scripts/download_vienna_full.py --zoom 18 --stitch
    python scripts/download_vienna_full.py --zoom 17 --dry-run
"""

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image
from tqdm import tqdm

# Full Vienna bounding box (covers entire city)
VIENNA_FULL_BBOX = {
    "min_lat": 48.10,
    "max_lat": 48.33,
    "min_lon": 16.17,
    "max_lon": 16.58,
    "description": "Full Vienna city extent",
}

# Output directory pattern: vienna_full_z{zoom}_{timestamp}
DEFAULT_OUTPUT_BASE = Path("data/raw/vienna")


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates at given zoom level (Web Mercator)."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def estimate_tiles(zoom: int) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
    """Estimate number of tiles for full Vienna at given zoom."""
    bbox = VIENNA_FULL_BBOX
    x_min, y_max = latlon_to_tile(bbox["min_lat"], bbox["min_lon"], zoom)
    x_max, y_min = latlon_to_tile(bbox["max_lat"], bbox["max_lon"], zoom)

    n_tiles_x = x_max - x_min + 1
    n_tiles_y = y_max - y_min + 1
    total = n_tiles_x * n_tiles_y

    return total, (x_min, x_max), (y_min, y_max)


def get_tile_url(
    x: int, y: int, zoom: int, layer: str = "lb", style: str = "farbe"
) -> str:
    """Generate WMTS tile URL for Vienna Orthofoto."""
    return f"https://mapsneu.wien.gv.at/wmts/{layer}/{style}/google3857/{zoom}/{y}/{x}.jpeg"


def download_tile(
    x: int, y: int, zoom: int, output_dir: Path, layer: str = "lb"
) -> Tuple[int, int, bool, str]:
    """Download a single tile.

    Returns:
        Tuple of (x, y, success, status) where status is 'downloaded', 'skipped', or error message
    """
    url = get_tile_url(x, y, zoom, layer)
    output_path = output_dir / f"tile_x{x}_y{y}.jpg"

    # Skip if already downloaded
    if output_path.exists():
        return x, y, True, "skipped"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return x, y, True, "downloaded"
    except Exception as e:
        return x, y, False, str(e)


def download_tiles_bbox(
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    zoom: int,
    output_dir: Path,
    layer: str = "lb",
    max_workers: int = 10,
) -> dict:
    """Download all tiles within tile coordinate ranges.

    Returns:
        Statistics dict with counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    x_min, x_max = x_range
    y_min, y_max = y_range

    # Build tile list
    tiles_to_download = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles_to_download.append((x, y))

    total_tiles = len(tiles_to_download)
    print(f"Total tiles to process: {total_tiles}")
    print(f"Tile range: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")
    print(f"Output directory: {output_dir}")

    # Download tiles in parallel
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "errors": []}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_tile, x, y, zoom, output_dir, layer): (x, y)
            for x, y in tiles_to_download
        }

        with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
            for future in as_completed(futures):
                x, y, success, status = future.result()
                if success:
                    if status == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["downloaded"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append(f"({x}, {y}): {status}")
                pbar.update(1)

    return stats


def stitch_tiles(
    tile_dir: Path,
    output_path: Path,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
) -> bool:
    """Stitch tiles into a single image."""
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Standard tile size for Web Mercator
    tile_size = 256
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size

    # Check if image would be too large (> 32k pixels in either dimension)
    if width > 32000 or height > 32000:
        print(f"Warning: Stitched image would be {width}x{height} pixels")
        print("This may be too large for some image viewers/software.")
        response = input("Continue with stitching? [y/N]: ")
        if response.lower() != "y":
            return False

    print(f"Stitching tiles into {width}x{height} image...")
    stitched = Image.new("RGB", (width, height))

    missing_tiles = 0
    for x in tqdm(range(x_min, x_max + 1), desc="Stitching rows"):
        for y in range(y_min, y_max + 1):
            tile_path = tile_dir / f"tile_x{x}_y{y}.jpg"
            if tile_path.exists():
                tile = Image.open(tile_path)
                stitched.paste(tile, ((x - x_min) * tile_size, (y - y_min) * tile_size))
            else:
                missing_tiles += 1

    if missing_tiles > 0:
        print(f"Warning: {missing_tiles} tiles were missing during stitching")

    stitched.save(output_path, "JPEG", quality=95)
    print(f"Saved stitched image to {output_path}")
    return True


def create_metadata_file(output_dir: Path, zoom: int, layer: str, stats: dict) -> None:
    """Create metadata file documenting the download."""
    metadata_path = output_dir / "metadata.txt"

    bbox = VIENNA_FULL_BBOX
    total, x_range, y_range = estimate_tiles(zoom)

    content = f"""Vienna Full Orthofoto Download
==============================
Date: {datetime.now().isoformat()}
Description: {bbox["description"]}

Bounding Box:
  Min Lat: {bbox["min_lat"]}
  Max Lat: {bbox["max_lat"]}
  Min Lon: {bbox["min_lon"]}
  Max Lon: {bbox["max_lon"]}

Download Parameters:
  Zoom Level: {zoom}
  Layer: {layer}
  Tile Range X: [{x_range[0]}, {x_range[1]}]
  Tile Range Y: [{y_range[0]}, {y_range[1]}]

Statistics:
  Total Tiles: {total}
  Downloaded: {stats["downloaded"]}
  Skipped (existing): {stats["skipped"]}
  Failed: {stats["failed"]}

License: Data from Stadt Wien under CC BY 4.0
Attribution: "Datenquelle: Stadt Wien – data.wien.gv.at"
"""

    with open(metadata_path, "w") as f:
        f.write(content)

    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download full Vienna Orthofoto tiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Estimated download sizes:
  Zoom 16: ~4,500 tiles (~280 MB)   - City overview
  Zoom 17: ~18,000 tiles (~1.1 GB)  - Recommended for large area
  Zoom 18: ~72,000 tiles (~4.5 GB)  - High detail
  Zoom 19: ~288,000 tiles (~18 GB)  - Very high detail

Examples:
  # Estimate tiles without downloading
  python scripts/download_vienna_full.py --zoom 17 --dry-run

  # Download at zoom 17 (good balance of detail and size)
  python scripts/download_vienna_full.py --zoom 17

  # Download and stitch (only for zoom 16-17, larger may fail)
  python scripts/download_vienna_full.py --zoom 16 --stitch

License: Data from Stadt Wien under CC BY 4.0
Attribution: "Datenquelle: Stadt Wien – data.wien.gv.at"
        """,
    )

    parser.add_argument(
        "--zoom",
        type=int,
        default=17,
        choices=[14, 15, 16, 17, 18, 19, 20],
        help="Zoom level (default: 17)",
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
        default=None,
        help="Override output directory (default: data/raw/vienna/orthofoto_full_z{zoom})",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only estimate tiles, don't download",
    )

    args = parser.parse_args()

    # Calculate tile estimates
    total_tiles, x_range, y_range = estimate_tiles(args.zoom)
    est_size_mb = total_tiles * 65 / 1024  # ~65KB per tile average

    bbox = VIENNA_FULL_BBOX
    print("=" * 60)
    print("Vienna Full Orthofoto Download")
    print("=" * 60)
    print(f"Area: {bbox['description']}")
    print(
        f"Bounding Box: ({bbox['min_lat']}, {bbox['min_lon']}) to ({bbox['max_lat']}, {bbox['max_lon']})"
    )
    print(f"Zoom Level: {args.zoom}")
    print(f"Layer: {args.layer}")
    print(f"Estimated Tiles: {total_tiles:,}")
    print(f"Estimated Size: {est_size_mb:.1f} MB ({est_size_mb / 1024:.2f} GB)")
    print(f"Tile Range: X=[{x_range[0]}, {x_range[1]}], Y=[{y_range[0]}, {y_range[1]}]")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] No files will be downloaded.")
        return

    # Confirm for large downloads
    if total_tiles > 10000:
        print(
            f"\nWarning: This will download {total_tiles:,} tiles ({est_size_mb / 1024:.1f} GB)"
        )
        response = input("Continue? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Set output directory - separate from existing data
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use descriptive name: orthofoto_full_z{zoom}
        output_dir = DEFAULT_OUTPUT_BASE / f"orthofoto_full_z{args.zoom}"

    print(f"\nOutput directory: {output_dir}")
    print("(Existing data in data/raw/vienna/orthofoto/ will NOT be modified)\n")

    # Download tiles
    stats = download_tiles_bbox(
        x_range,
        y_range,
        args.zoom,
        output_dir,
        args.layer,
        args.workers,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Downloaded: {stats['downloaded']:,} new tiles")
    print(f"Skipped: {stats['skipped']:,} existing tiles")
    print(f"Failed: {stats['failed']:,} tiles")

    if stats["errors"]:
        print("\nFirst 5 errors:")
        for err in stats["errors"][:5]:
            print(f"  {err}")

    # Save metadata
    create_metadata_file(output_dir, args.zoom, args.layer, stats)

    # Optional stitching
    if args.stitch:
        output_path = output_dir / f"vienna_full_z{args.zoom}_stitched.jpg"
        stitch_tiles(output_dir, output_path, x_range, y_range)

    print("\nDone!")
    print(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
