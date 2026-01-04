#!/usr/bin/env python
"""
Compute and save self-similarity matrix for satellite imagery.

This script computes the (N×K) × (N×K) self-similarity matrix where N×K is the
number of DINOv3 patches in the image. The similarity matrix can be pre-computed
once and reused for fast interactive similarity search without recomputing
patch-to-patch similarities.

Usage:
    # From embeddings file
    python scripts/compute_similarity_matrix.py \\
        --embeddings data/vienna_embeddings.npz \\
        --output data/vienna_similarity_matrix.npy

    # From image file (will compute embeddings first)
    python scripts/compute_similarity_matrix.py \\
        --image data/vienna.jpg \\
        --output data/vienna_similarity_matrix.npy
"""

import argparse
import time
from pathlib import Path

import numpy as np

from satfinder.precompute import load_precomputed_embeddings, precompute_embeddings
from satfinder.similarity import SimilarityEngine


def compute_similarity_matrix_from_embeddings(
    embeddings_path: Path,
    output_path: Path,
    device: str | None = None,
    batch_size: int = 512,
    verbose: bool = True,
) -> dict:
    """
    Compute similarity matrix from pre-computed embeddings.

    Args:
        embeddings_path: Path to .npz file with pre-computed embeddings
        output_path: Path to save similarity matrix (.npy file)
        device: Device to run computation on (None = auto-detect)
        batch_size: Batch size for computing similarity matrix (default: 512)
        verbose: Print progress messages

    Returns:
        Dictionary with metadata about the computation
    """
    total_start = time.time()

    def log(msg, **kwargs):
        if verbose:
            print(msg, **kwargs)

    log(f"Computing similarity matrix from embeddings: {embeddings_path}")

    # Step 1: Load pre-computed embeddings
    log("  [1/3] Loading embeddings...", end=" ", flush=True)
    load_start = time.time()
    data = load_precomputed_embeddings(embeddings_path)
    features = data["features"]
    image_shape = data["image_shape"]
    backbone = data["backbone"]
    log(f"done ({time.time() - load_start:.1f}s)")
    log(f"        Feature shape: {features.shape}")
    log(f"        Image shape: {image_shape}")
    log(f"        Backbone: {backbone}")

    # Step 2: Initialize SimilarityEngine and load features
    log("  [2/3] Initializing engine...", end=" ", flush=True)
    init_start = time.time()
    engine = SimilarityEngine(backbone=backbone, device=device)
    engine.load_precomputed_features(features=features, image_size=image_shape)
    log(f"done ({time.time() - init_start:.1f}s)")

    # Step 3: Compute similarity matrix (saves directly to disk for large matrices)
    log("  [3/3] Computing similarity matrix...")
    compute_start = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass output_path to compute_similarity_matrix for memory-mapped mode
    similarity_matrix = engine.compute_similarity_matrix(
        output_path=output_path, batch_size=batch_size
    )

    compute_time = time.time() - compute_start
    log(f"        Computation time: {compute_time:.1f}s")

    # Print summary
    size_mb = output_path.stat().st_size / (1024 * 1024)
    total_time = time.time() - total_start
    log(f"  Saved: {output_path}")
    log(f"  Size: {size_mb:.1f} MB | Total time: {total_time:.1f}s")

    # Get matrix shape from file (since similarity_matrix is None when using memmap)
    if similarity_matrix is None:
        # Load just the shape metadata without loading the full matrix
        with open(output_path, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
        matrix_shape = shape
    else:
        matrix_shape = similarity_matrix.shape

    return {
        "matrix_shape": matrix_shape,
        "file_size_mb": size_mb,
        "total_time_s": total_time,
    }


def compute_similarity_matrix_from_image(
    image_path: Path,
    output_path: Path,
    backbone: str = "dinov3_sat_large",
    device: str | None = None,
    tile_size: int = 512,
    batch_size: int = 4,
    sim_batch_size: int = 512,
    verbose: bool = True,
) -> dict:
    """
    Compute similarity matrix directly from image (will compute embeddings first).

    Args:
        image_path: Path to input image
        output_path: Path to save similarity matrix (.npy file)
        backbone: DINOv3 backbone model name
        device: Device to run computation on (None = auto-detect)
        tile_size: Tile size for large images
        batch_size: Batch size for processing tiles
        sim_batch_size: Batch size for computing similarity matrix (default: 512)
        verbose: Print progress messages

    Returns:
        Dictionary with metadata about the computation
    """
    total_start = time.time()

    def log(msg, **kwargs):
        if verbose:
            print(msg, **kwargs)

    log(f"Computing similarity matrix from image: {image_path}")
    log("  This will first compute embeddings, then the similarity matrix")

    # Step 1: Compute embeddings (save to temporary file)
    temp_embeddings_path = output_path.with_suffix(".embeddings.npz")
    log("\n[Stage 1/2] Computing embeddings...")
    embed_metadata = precompute_embeddings(
        image_path=image_path,
        output_path=temp_embeddings_path,
        backbone_name=backbone,
        device=device,
        tile_size=tile_size,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Step 2: Compute similarity matrix from embeddings
    log("\n[Stage 2/2] Computing similarity matrix...")
    sim_metadata = compute_similarity_matrix_from_embeddings(
        embeddings_path=temp_embeddings_path,
        output_path=output_path,
        device=device,
        batch_size=sim_batch_size,
        verbose=verbose,
    )

    # Optionally delete temp embeddings file
    # temp_embeddings_path.unlink()  # Uncomment to delete temp file

    total_time = time.time() - total_start
    log(f"\n  Total processing time: {total_time:.1f}s")

    return {
        **sim_metadata,
        "embeddings_file": str(temp_embeddings_path),
        "embeddings_size_mb": embed_metadata["file_size_mb"],
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute self-similarity matrix for satellite imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From pre-computed embeddings
  python scripts/compute_similarity_matrix.py \\
      --embeddings data/vienna_embeddings.npz \\
      --output data/vienna_similarity.npy

  # From image (will compute embeddings first)
  python scripts/compute_similarity_matrix.py \\
      --image data/vienna.jpg \\
      --output data/vienna_similarity.npy \\
      --backbone dinov3_sat_large \\
      --batch-size 8

  # Use GPU if available
  python scripts/compute_similarity_matrix.py \\
      --embeddings data/vienna_embeddings.npz \\
      --output data/vienna_similarity.npy \\
      --device cuda
        """,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--embeddings",
        type=Path,
        help="Path to pre-computed embeddings (.npz file)",
    )
    input_group.add_argument(
        "--image",
        type=Path,
        help="Path to input image (will compute embeddings first)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save similarity matrix (.npy file)",
    )

    # Model options
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

    # Processing options (only for --image mode)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size for large images (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing tiles (default: 4)",
    )
    parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=512,
        help="Batch size for computing similarity matrix (default: 512, reduce if OOM)",
    )

    # Verbosity
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Compute similarity matrix
    if args.embeddings:
        # From pre-computed embeddings
        metadata = compute_similarity_matrix_from_embeddings(
            embeddings_path=args.embeddings,
            output_path=args.output,
            device=args.device,
            batch_size=args.sim_batch_size,
            verbose=not args.quiet,
        )
    else:
        # From image (will compute embeddings first)
        metadata = compute_similarity_matrix_from_image(
            image_path=args.image,
            output_path=args.output,
            backbone=args.backbone,
            device=args.device,
            tile_size=args.tile_size,
            batch_size=args.batch_size,
            sim_batch_size=args.sim_batch_size,
            verbose=not args.quiet,
        )

    if not args.quiet:
        print("\n✅ Similarity matrix computation complete!")
        print(f"   Output: {args.output}")
        print(f"   Matrix shape: {metadata['matrix_shape']}")
        print(f"   File size: {metadata['file_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
