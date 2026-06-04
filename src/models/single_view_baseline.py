"""Single-view reconstruction baseline: frozen DINOv2 + point cloud head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .dinov2_encoder import DINOv2Encoder
from .point_cloud_head import build_point_cloud_head


class SingleViewBaseline(nn.Module):
    """
    Lower-bound baseline from the project spec:
    one RGB view -> frozen DINOv2 -> trainable point cloud decoder.
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
    ) -> None:
        super().__init__()
        self.encoder = DINOv2Encoder(
            dinov2_variant,
            freeze=freeze_backbone,
            feature_mode=feature_mode,
            unfreeze_last_blocks=unfreeze_last_blocks,
        )
        self.head = build_point_cloud_head(
            head_type=head_type,
            input_dim=self.encoder.out_dim,
            num_points=num_output_points,
            hidden_dim=hidden_dim,
        )

    def forward(self, images: torch.Tensor, poses: torch.Tensor | None = None) -> torch.Tensor:
        if images.ndim == 5:
            images = images[:, 0]
        features = self.encoder(images)
        return self.head(features)
