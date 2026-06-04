"""Multi-view reconstruction baselines."""

from __future__ import annotations

import torch
import torch.nn as nn

from .dinov2_encoder import DINOv2Encoder
from .point_cloud_head import build_point_cloud_head


class MultiViewBaseline(nn.Module):
    """
    Multi-view baseline: encode each view with DINOv2, fuse features, decode to point cloud.
    Supports naive mean pooling (pose-agnostic) from the project spec.
    """

    def __init__(
        self,
        dinov2_variant: str = "dinov2_vitb14",
        num_output_points: int = 2048,
        hidden_dim: int = 512,
        freeze_backbone: bool = True,
        feature_mode: str = "cls",
        head_type: str = "mlp",
        unfreeze_last_blocks: int = 0,
        fusion: str = "mean_pool",
        num_views: int = 2,
        fusion_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.fusion = fusion
        self.encoder = DINOv2Encoder(
            dinov2_variant,
            freeze=freeze_backbone,
            feature_mode=feature_mode,
            unfreeze_last_blocks=unfreeze_last_blocks,
        )

        head_input_dim = self.encoder.out_dim
        if fusion == "concat":
            head_input_dim *= num_views

        self.head = build_point_cloud_head(
            head_type=head_type,
            input_dim=head_input_dim,
            num_points=num_output_points,
            hidden_dim=hidden_dim,
        )

        if fusion == "weighted_mean":
            weights = fusion_weights or [1.0 / num_views] * num_views
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            weight_tensor = weight_tensor / weight_tensor.sum()
            self.register_buffer("fusion_weights", weight_tensor)

    def encode_views(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views = images.shape[:2]
        flat = images.reshape(batch_size * num_views, *images.shape[2:])
        features = self.encoder(flat)
        return features.view(batch_size, num_views, -1)

    def fuse_features(self, view_features: torch.Tensor) -> torch.Tensor:
        if self.fusion == "mean_pool":
            return view_features.mean(dim=1)
        if self.fusion == "weighted_mean":
            weights = self.fusion_weights.view(1, -1, 1)
            return (view_features * weights).sum(dim=1)
        if self.fusion == "concat":
            batch_size = view_features.shape[0]
            return view_features.reshape(batch_size, -1)
        if self.fusion == "view0":
            return view_features[:, 0]
        raise ValueError(f"Unknown fusion: {self.fusion}")

    def forward(self, images: torch.Tensor, poses: torch.Tensor | None = None) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        view_features = self.encode_views(images)
        fused = self.fuse_features(view_features)
        return self.head(fused)
