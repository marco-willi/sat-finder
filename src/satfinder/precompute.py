"""Pre-computation utilities for DINOv3 embeddings on satellite imagery."""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .dataset import ImageTileDataset
from .features import DINOBackbone
from .similarity import get_normalization_transform


def precompute_embeddings(
    image_path: Path,
    output_path: Path,
    backbone_name: str = "dinov3_sat_large",
    device: str | None = None,
    tile_size: int = 512,
    batch_size: int = 4,
    verbose: bool = True,
) -> dict:
    """
    Pre-compute and save DINOv3 embeddings for a single satellite image.

    For images larger than tile_size × tile_size, the image is tiled and
    processed in batches for efficiency. Features are assembled into a single
    dense feature map and saved as a compressed .npz file.

    Args:
        image_path: Path to input image (any PIL-supported format)
        output_path: Path to save .npz file with embeddings
        backbone_name: DINOv3 backbone model name (default: "dinov3_sat_large")
        device: Device to run inference on (None = auto-detect)
        tile_size: Size of square tiles for large images (default: 512)
        batch_size: Batch size for processing tiles (default: 4)
        verbose: Print progress messages (default: True)

    Returns:
        Dictionary with metadata:
            - features_shape: Shape of saved feature tensor (H', W', D)
            - image_shape: Original image shape (H, W)
            - file_size_mb: Size of output file in MB
            - total_time_s: Total processing time in seconds

    Example:
        >>> from pathlib import Path
        >>> from satfinder.precompute import precompute_embeddings
        >>>
        >>> metadata = precompute_embeddings(
        ...     image_path=Path("data/vienna.jpg"),
        ...     output_path=Path("data/vienna_embeddings.npz"),
        ...     batch_size=8
        ... )
        >>> print(f"Saved features: {metadata['features_shape']}")
    """
    total_start = time.time()

    def log(msg, **kwargs):
        if verbose:
            print(msg, **kwargs)

    log(f"Processing: {image_path}")

    # Step 1: Load image
    log("  [1/4] Loading image...", end=" ", flush=True)
    load_start = time.time()
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    log(f"done ({time.time() - load_start:.1f}s)")
    log(f"        Image size: {image_np.shape[1]}x{image_np.shape[0]}")

    # Step 2: Initialize backbone
    log("  [2/4] Loading backbone...", end=" ", flush=True)
    backbone_start = time.time()
    backbone = DINOBackbone(backbone=backbone_name, device=device)
    log(f"done ({time.time() - backbone_start:.1f}s)")
    log(f"        Device: {backbone.device}")

    # Get normalization transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            get_normalization_transform(backbone_name),
        ]
    )

    # Step 3: Extract features
    height, width = image_np.shape[:2]
    needs_tiling = height > tile_size or width > tile_size

    log("  [3/4] Extracting features...", end=" ", flush=True)
    extract_start = time.time()

    if needs_tiling:
        features_np = _extract_features_tiled(
            image_np=image_np,
            backbone=backbone,
            transform=transform,
            tile_size=tile_size,
            batch_size=batch_size,
            verbose=verbose,
        )
    else:
        features_np = _extract_features_whole(
            image_np=image_np,
            backbone=backbone,
            transform=transform,
        )
        log(f"done ({time.time() - extract_start:.1f}s)")

    log(f"        Feature shape: {features_np.shape}")

    # Step 4: Save embeddings with metadata
    log("  [4/4] Saving embeddings...", end=" ", flush=True)
    save_start = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        features=features_np,
        image_shape=image_np.shape[:2],  # (H, W)
        backbone=backbone_name,
        patch_size=backbone.get_patch_size(),
        feature_dim=backbone.get_feature_dim(),
        source_image=str(image_path),
    )
    log(f"done ({time.time() - save_start:.1f}s)")

    # Print summary
    size_mb = output_path.stat().st_size / (1024 * 1024)
    total_time = time.time() - total_start
    log(f"  Saved: {output_path}")
    log(f"  Size: {size_mb:.1f} MB | Total time: {total_time:.1f}s")

    return {
        "features_shape": features_np.shape,
        "image_shape": image_np.shape[:2],
        "file_size_mb": size_mb,
        "total_time_s": total_time,
    }


def _extract_features_whole(
    image_np: np.ndarray,
    backbone: DINOBackbone,
    transform: transforms.Compose,
) -> np.ndarray:
    """Extract features from entire image at once (for small images).

    Args:
        image_np: Image as numpy array (H, W, C)
        backbone: DINOBackbone instance
        transform: Preprocessing transform

    Returns:
        Features as numpy array (H', W', D)
    """
    tensor = transform(image_np).unsqueeze(0).to(backbone.device)

    with torch.no_grad():
        features = backbone(tensor)  # [1, H', W', D]

    return features[0].cpu().numpy()  # [H', W', D]


def _extract_features_tiled(
    image_np: np.ndarray,
    backbone: DINOBackbone,
    transform: transforms.Compose,
    tile_size: int,
    batch_size: int,
    verbose: bool = True,
) -> np.ndarray:
    """Extract features from large image using tiling strategy.

    Args:
        image_np: Image as numpy array (H, W, C)
        backbone: DINOBackbone instance
        transform: Preprocessing transform
        tile_size: Size of square tiles
        batch_size: Batch size for processing tiles
        verbose: Print progress messages

    Returns:
        Features as numpy array (H', W', D)
    """

    def log(msg, **kwargs):
        if verbose:
            print(msg, **kwargs)

    height, width = image_np.shape[:2]

    # Create dataset and dataloader for tiles
    log(f"\n        Image is large, tiling into {tile_size}x{tile_size} patches...")
    dataset = ImageTileDataset(image_np, tile_size=tile_size)
    log(
        f"        Created {len(dataset)} tiles ({dataset.n_tiles_h}x{dataset.n_tiles_w})"
    )

    # Calculate expected feature map dimensions for a single tile
    with torch.no_grad():
        sample_tile = dataset[0]["tile"]
        sample_tensor = transform(sample_tile).unsqueeze(0).to(backbone.device)
        sample_features = backbone(sample_tensor)
        tile_feat_h, tile_feat_w = sample_features.shape[1:3]

    # Initialize full feature map based on actual image dimensions
    # NOT based on number of tiles (which may include padding)
    patch_size = backbone.get_patch_size()
    full_feat_h = height // patch_size
    full_feat_w = width // patch_size
    feature_dim = backbone.get_feature_dim()
    features_np = np.zeros((full_feat_h, full_feat_w, feature_dim), dtype=np.float32)

    # Process tiles in batches
    log(f"        Processing tiles in batches of {batch_size}...", flush=True)
    for i in range(0, len(dataset), batch_size):
        batch_end = min(i + batch_size, len(dataset))
        batch_tiles = [dataset[j] for j in range(i, batch_end)]

        # Stack tiles into batch
        tiles_np = np.stack([item["tile"] for item in batch_tiles])
        tiles_tensor = torch.stack([transform(tile) for tile in tiles_np]).to(
            backbone.device
        )

        # Extract features for batch
        with torch.no_grad():
            batch_features = backbone(tiles_tensor)  # [B, H', W', D]

        # Place features in correct position
        for j, item in enumerate(batch_tiles):
            y_start = item["y_start"]
            x_start = item["x_start"]
            y_end = item["y_end"]
            x_end = item["x_end"]

            # Calculate feature coordinates based on actual pixel coordinates
            feat_y_start = y_start // patch_size
            feat_x_start = x_start // patch_size
            feat_y_end = y_end // patch_size
            feat_x_end = x_end // patch_size

            # Extract only the valid (non-padded) portion of features
            feat_h = feat_y_end - feat_y_start
            feat_w = feat_x_end - feat_x_start

            features_np[feat_y_start:feat_y_end, feat_x_start:feat_x_end] = (
                batch_features[j][:feat_h, :feat_w].cpu().numpy()
            )

        log(f"        Processed tiles {i + 1}-{batch_end}/{len(dataset)}", flush=True)

    log(f"  done ({time.time():.1f}s)")
    return features_np


def load_precomputed_embeddings(embeddings_path: Path) -> dict:
    """
    Load pre-computed embeddings from .npz file.

    Args:
        embeddings_path: Path to .npz file created by precompute_embeddings()

    Returns:
        Dictionary with keys:
            - features: Feature tensor as numpy array (H', W', D)
            - image_shape: Original image shape (H, W)
            - backbone: Backbone name used
            - patch_size: Patch size (typically 16)
            - feature_dim: Feature dimension (typically 1024)
            - source_image: Path to original image (if saved)

    Example:
        >>> from pathlib import Path
        >>> from satfinder.precompute import load_precomputed_embeddings
        >>>
        >>> data = load_precomputed_embeddings(Path("data/vienna_embeddings.npz"))
        >>> features = data['features']
        >>> print(f"Feature shape: {features.shape}")
    """
    data = np.load(embeddings_path)

    return {
        "features": data["features"],
        "image_shape": tuple(data["image_shape"]),
        "backbone": str(data["backbone"]),
        "patch_size": int(data["patch_size"]),
        "feature_dim": int(data["feature_dim"]),
        "source_image": str(data.get("source_image", "")),
    }
