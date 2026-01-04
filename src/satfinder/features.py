"""
DINOv3 backbone wrapper for feature extraction.

Supports DINOv3 backbones via timm. Model definitions are loaded from
config/models.yaml.
"""

import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)


def _load_backbone_config() -> dict:
    """Load backbone definitions from config/models.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config["backbones"]


# Load config once at module import
BACKBONE_CONFIG = _load_backbone_config()


def get_available_backbones() -> list[str]:
    """Get list of available backbone names."""
    return list(BACKBONE_CONFIG.keys())


def get_backbone_info(backbone_name: str) -> dict:
    """
    Get backbone configuration by name.

    Args:
        backbone_name: Name of the backbone (e.g., 'dinov3_base', 'dinov3_sat_large')

    Returns:
        Dict with timm_name, feature_dim, patch_size

    Raises:
        ValueError: If backbone name is not found
    """
    if backbone_name not in BACKBONE_CONFIG:
        available = ", ".join(get_available_backbones())
        raise ValueError(f"Unknown backbone: '{backbone_name}'. Available: {available}")
    return BACKBONE_CONFIG[backbone_name]


class DINOBackbone(nn.Module):
    """
    DINOv3 backbone for dense feature extraction.

    Wraps timm DINOv3 models and extracts dense features suitable for
    dense prediction tasks like segmentation and similarity search.
    """

    def __init__(
        self,
        backbone: str = "dinov3_sat_large",
        pretrained: bool = True,
        freeze: bool = True,
        device: str | torch.device | None = None,
    ):
        """
        Initialize DINO backbone.

        Args:
            backbone: Backbone name from config/models.yaml
                      (e.g., 'dinov3_base', 'dinov3_sat_large')
            pretrained: Load pretrained weights
            freeze: Freeze backbone weights
            device: Device to run inference on. If None, auto-selects GPU/CPU.
        """
        super().__init__()

        # Auto-select device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Get backbone config
        config = get_backbone_info(backbone)
        self.backbone_name = backbone
        self.timm_name = config["timm_name"]
        self.feature_dim = config["feature_dim"]
        self.patch_size = config["patch_size"]
        self.output_stride = self.patch_size

        logger.info(
            f"Loading DINO backbone: {self.timm_name} "
            f"(feature_dim={self.feature_dim}, patch_size={self.patch_size})"
        )

        # Create model without classification head
        logger.info("Creating timm model (this may download weights on first run)...")
        self.backbone = timm.create_model(
            self.timm_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="",  # No global pooling - keep spatial dims
            dynamic_img_size=True,  # Allow variable input sizes
        )
        logger.info("timm model created successfully")

        # Move to device and freeze if requested
        logger.info(f"Moving model to {self.device}...")
        self.backbone.to(self.device)
        logger.info("Model moved to device")
        if freeze:
            logger.info("Freezing backbone weights...")
            self.freeze()
            logger.info("Backbone frozen")

    def freeze(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze(self, layers: int = -1) -> None:
        """
        Unfreeze backbone parameters.

        Args:
            layers: Number of transformer blocks to unfreeze from the end.
                   -1 means unfreeze all.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        if layers > 0:
            # Re-freeze early layers
            blocks = self.backbone.blocks
            n_blocks = len(blocks)
            for i, block in enumerate(blocks):
                if i < n_blocks - layers:
                    for param in block.parameters():
                        param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract dense features from input image.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Feature tensor [B, H/P, W/P, D] where P is patch_size
        """
        B, C, H, W = x.shape
        logger.info(f"DINOBackbone.forward: input shape=[{B}, {C}, {H}, {W}]")

        # Move to device if needed
        if x.device != self.device:
            logger.info(f"Moving input from {x.device} to {self.device}")
            x = x.to(self.device)

        # Get features from backbone: [B, N, C] where N = num_patches + num_registers
        logger.info("Running backbone.forward_features()...")
        features = self.backbone.forward_features(x)
        logger.info(f"backbone.forward_features() complete: shape={features.shape}")

        # For ViT, reshape patch tokens to spatial grid
        # Skip CLS token and any register tokens
        if hasattr(self.backbone, "num_prefix_tokens"):
            num_prefix = self.backbone.num_prefix_tokens
        else:
            num_prefix = 1  # Just CLS token

        # Remove prefix tokens
        patch_features = features[:, num_prefix:, :]  # [B, N_patches, C]

        # Reshape to spatial grid
        h = H // self.output_stride
        w = W // self.output_stride

        # Handle case where actual patches don't match expected
        n_patches = patch_features.shape[1]
        expected_patches = h * w

        if n_patches != expected_patches:
            # Interpolate to expected size
            patch_features = patch_features.transpose(1, 2)  # [B, C, N]
            patch_features = patch_features.unsqueeze(-1)  # [B, C, N, 1]
            patch_features = F.interpolate(
                patch_features, size=(expected_patches, 1), mode="bilinear"
            )
            patch_features = patch_features.squeeze(-1).transpose(1, 2)

        # Reshape to [B, H', W', C] for similarity search
        spatial_features = patch_features.reshape(B, h, w, -1)
        logger.info(
            f"DINOBackbone.forward complete: output shape={spatial_features.shape}"
        )

        return spatial_features

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.feature_dim

    def get_patch_size(self) -> int:
        """Get patch size (output stride)."""
        return self.patch_size
