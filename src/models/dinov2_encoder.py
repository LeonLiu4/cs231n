"""DINOv2 feature extractor with optional partial unfreezing."""

from __future__ import annotations

import torch
import torch.nn as nn


class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        variant: str = "dinov2_vitb14",
        freeze: bool = True,
        feature_mode: str = "cls",
        unfreeze_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", variant)
        self.embed_dim = self.model.embed_dim
        self.feature_mode = feature_mode
        self.unfreeze_last_blocks = unfreeze_last_blocks

        for param in self.model.parameters():
            param.requires_grad = False

        if unfreeze_last_blocks > 0:
            for block in self.model.blocks[-unfreeze_last_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        if freeze and unfreeze_last_blocks == 0:
            self.model.eval()

        if feature_mode == "cls_patch_mean":
            self.out_dim = self.embed_dim * 2
        elif feature_mode == "cls":
            self.out_dim = self.embed_dim
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

    @property
    def has_trainable_backbone(self) -> bool:
        return any(p.requires_grad for p in self.model.parameters())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "cls":
            if not self.has_trainable_backbone:
                with torch.no_grad():
                    return self.model(images)
            return self.model(images)

        if not self.has_trainable_backbone:
            with torch.no_grad():
                features = self.model.forward_features(images)
        else:
            features = self.model.forward_features(images)

        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        if self.feature_mode == "cls_patch_mean":
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
        raise ValueError(f"Unknown feature_mode: {self.feature_mode}")
