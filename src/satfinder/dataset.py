"""Dataset utilities for satellite imagery processing."""

import numpy as np
from torch.utils.data import Dataset


class ImageTileDataset(Dataset):
    """Dataset for tiling large images into fixed-size patches.

    This is useful for processing large satellite imagery that doesn't fit
    in memory or VRAM. The dataset tiles the image into overlapping or
    non-overlapping patches of a specified size.

    Example:
        >>> image = np.random.rand(2048, 2048, 3).astype(np.uint8)
        >>> dataset = ImageTileDataset(image, tile_size=512)
        >>> print(len(dataset))  # Number of tiles
        16
        >>> tile_data = dataset[0]
        >>> print(tile_data['tile'].shape)
        (512, 512, 3)
    """

    def __init__(self, image: np.ndarray, tile_size: int = 512):
        """
        Initialize the dataset with an image to tile.

        Args:
            image: Input image as numpy array (H, W, C)
            tile_size: Size of square tiles (default: 512)
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image (H, W, C), got shape {image.shape}")

        self.image = image
        self.tile_size = tile_size
        self.height, self.width = image.shape[:2]

        # Calculate number of tiles in each dimension (ceiling division)
        self.n_tiles_h = (self.height + tile_size - 1) // tile_size
        self.n_tiles_w = (self.width + tile_size - 1) // tile_size

    def __len__(self):
        """Return total number of tiles."""
        return self.n_tiles_h * self.n_tiles_w

    def __getitem__(self, idx):
        """
        Get a tile by index.

        Args:
            idx: Tile index (row-major order)

        Returns:
            Dictionary containing:
                - tile: Tile image array (tile_size, tile_size, C), padded if edge tile
                - tile_idx: Original index
                - tile_row: Row position in tile grid
                - tile_col: Column position in tile grid
                - y_start, x_start: Top-left coordinates in original image
                - y_end, x_end: Bottom-right coordinates in original image
        """
        # Calculate tile position in grid (row-major order)
        tile_row = idx // self.n_tiles_w
        tile_col = idx % self.n_tiles_w

        # Calculate crop coordinates in original image
        y_start = tile_row * self.tile_size
        x_start = tile_col * self.tile_size
        y_end = min(y_start + self.tile_size, self.height)
        x_end = min(x_start + self.tile_size, self.width)

        # Extract tile from original image
        tile = self.image[y_start:y_end, x_start:x_end]

        # Pad edge tiles to tile_size × tile_size (zero padding)
        if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
            padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=tile.dtype)
            padded[: tile.shape[0], : tile.shape[1]] = tile
            tile = padded

        return {
            "tile": tile,
            "tile_idx": idx,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "y_start": y_start,
            "x_start": x_start,
            "y_end": y_end,
            "x_end": x_end,
        }

    def get_tile_grid_shape(self):
        """
        Get the shape of the tile grid.

        Returns:
            Tuple of (n_tiles_h, n_tiles_w)
        """
        return (self.n_tiles_h, self.n_tiles_w)

    def get_feature_grid_shape(self, patch_size: int):
        """
        Calculate the feature grid shape after DINOv3 processing.

        Args:
            patch_size: DINOv3 patch size (typically 16)

        Returns:
            Tuple of (feature_h, feature_w) for the full image
        """
        # Each tile becomes (tile_size // patch_size)² features
        tile_feat_h = self.tile_size // patch_size
        tile_feat_w = self.tile_size // patch_size

        # Total features = n_tiles * features_per_tile
        full_feat_h = self.n_tiles_h * tile_feat_h
        full_feat_w = self.n_tiles_w * tile_feat_w

        return (full_feat_h, full_feat_w)
