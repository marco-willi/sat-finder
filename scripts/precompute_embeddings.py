#!/usr/bin/env python3
"""
Pre-compute DINOv3 embeddings for satellite imagery.

Extracts dense feature embeddings from images and saves them as .npz files
for fast loading during interactive sessions.

Usage:
    python scripts/precompute_embeddings.py
    python scripts/precompute_embeddings.py --backbone dinov3_base
    python scripts/precompute_embeddings.py --input data/raw/custom/image.jpg

Note:
    The core functionality has been moved to satfinder.precompute module.
    This script is a CLI wrapper around that module for convenience.
"""

import argparse
from pathlib import Path

from satfinder.precompute import precompute_embeddings


def find_images(data_dir: Path, pattern: str = "vienna_z18_stitched.jpg") -> list[Path]:
    """Find all matching images in data directory."""
    return list(data_dir.glob(f"**/{pattern}"))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute DINOv3 embeddings for satellite imagery"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to input image (if not specified, searches for default patterns)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save embeddings .npz (auto-generated if not specified)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov3_sat_large",
        help="Backbone model name (default: dinov3_sat_large)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda, cpu, or None for auto-detect)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Size of tiles for large images (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing tiles (default: 4)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory to search for images (default: data/)",
    )

    args = parser.parse_args()

    # Find images to process
    if args.input:
        image_paths = [args.input]
    else:
        # Search for default patterns
        print(f"Searching for images in {args.data_dir}...")
        image_paths = find_images(args.data_dir)
        if not image_paths:
            print(f"No images found in {args.data_dir}")
            print("Please specify --input or ensure data directory contains images")
            return

    # Process each image
    for image_path in image_paths:
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            continue

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Auto-generate output path: same directory, add _embeddings.npz
            output_path = image_path.parent / f"{image_path.stem}_embeddings.npz"

        # Pre-compute embeddings
        precompute_embeddings(
            image_path=image_path,
            output_path=output_path,
            backbone_name=args.backbone,
            device=args.device,
            tile_size=args.tile_size,
            batch_size=args.batch_size,
            verbose=True,
        )

        print()  # Empty line between images


if __name__ == "__main__":
    main()
