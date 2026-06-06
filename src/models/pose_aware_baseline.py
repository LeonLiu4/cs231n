"""Pose-aware multi-view reconstruction models.

These variants share the frozen DINOv2 encoder, fusion, and point-cloud head
of :class:`MultiViewBaseline`, but inject per-view positional/geometry
information between view encoding and fusion. They implement the baselines in
the writeup's "Baselines" table plus the proposed geometry-aware model:

    pos_embedding="view_id"        -> learned per-view-index embedding
    pos_embedding="camera_pose"    -> MLP over the raw 7-D camera pose vector
    pos_embedding="geometry_aware" -> Fourier-encoded camera geometry (proposed)

The plain "2D positional embedding" / "multi-view" baselines need no injection
(DINOv2 already carries ViT 2-D patch positions), so they use
:class:`MultiViewBaseline` directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .multi_view_baseline import MultiViewBaseline

POSE_VECTOR_DIM = 7  # [elev_deg, azim_deg, distance, eye_x, eye_y, eye_z, 1.0]


def _normalize_pose(poses: torch.Tensor) -> torch.Tensor:
    """Scale the raw 7-D pose vector to roughly unit range for stable MLPs."""
    elev = poses[..., 0:1] / 90.0
    azim = poses[..., 1:2] / 360.0
    distance = poses[..., 2:3] / 2.0
    eye = poses[..., 3:6] / 2.0
    flag = poses[..., 6:7]
    return torch.cat([elev, azim, distance, eye, flag], dim=-1)


def _geometry_features(poses: torch.Tensor, num_freqs: int = 6) -> torch.Tensor:
    """Build a geometry descriptor from a camera pose vector.

    Encodes elevation/azimuth as sin/cos (angle-aware, wrap-safe), the unit
    viewing direction (from eye position), distance, and a Fourier feature
    expansion of the eye position. Returns ``[..., F]``.
    """
    elev = torch.deg2rad(poses[..., 0])
    azim = torch.deg2rad(poses[..., 1])
    distance = poses[..., 2:3] / 2.0
    eye = poses[..., 3:6]

    direction = eye / (eye.norm(dim=-1, keepdim=True) + 1e-6)

    angle_feats = torch.stack(
        [torch.sin(elev), torch.cos(elev), torch.sin(azim), torch.cos(azim)],
        dim=-1,
    )

    fourier = []
    for i in range(num_freqs):
        freq = 2.0**i * math.pi
        fourier.append(torch.sin(freq * eye))
        fourier.append(torch.cos(freq * eye))
    fourier_feats = torch.cat(fourier, dim=-1)

    return torch.cat([angle_feats, direction, distance, fourier_feats], dim=-1)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoseAwareMultiViewBaseline(MultiViewBaseline):
    """Multi-view baseline that adds a per-view positional/geometry embedding.

    The embedding is projected to the encoder feature dimension and added to
    each view's feature vector before fusion, so the fusion/head input
    dimensions are unchanged relative to :class:`MultiViewBaseline`.
    """

    def __init__(
        self,
        *,
        pos_embedding: str = "geometry_aware",
        pose_embed_hidden: int = 128,
        geometry_num_freqs: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pos_embedding = pos_embedding
        self.geometry_num_freqs = geometry_num_freqs

        embed_dim = self.encoder.out_dim

        if pos_embedding == "view_id":
            self.view_embedding = nn.Embedding(self.num_views, embed_dim)
        elif pos_embedding == "camera_pose":
            self.pose_encoder = _MLP(POSE_VECTOR_DIM, pose_embed_hidden, embed_dim)
        elif pos_embedding == "geometry_aware":
            # angle(4) + direction(3) + distance(1) + fourier(6 * num_freqs)
            geo_dim = 4 + 3 + 1 + 6 * geometry_num_freqs
            self.geometry_encoder = _MLP(geo_dim, pose_embed_hidden, embed_dim)
        else:
            raise ValueError(
                f"Unknown pos_embedding: {pos_embedding!r}. Use one of "
                "'view_id', 'camera_pose', 'geometry_aware'."
            )

    def _positional_embedding(
        self, view_features: torch.Tensor, poses: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size, num_views = view_features.shape[:2]

        if self.pos_embedding == "view_id":
            idx = torch.arange(num_views, device=view_features.device)
            idx = idx.unsqueeze(0).expand(batch_size, num_views)
            return self.view_embedding(idx)

        if poses is None:
            raise ValueError(
                f"pos_embedding={self.pos_embedding!r} requires poses, got None."
            )

        if self.pos_embedding == "camera_pose":
            return self.pose_encoder(_normalize_pose(poses))

        geo = _geometry_features(poses, num_freqs=self.geometry_num_freqs)
        return self.geometry_encoder(geo)

    def forward(
        self, images: torch.Tensor, poses: torch.Tensor | None = None
    ) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        view_features = self.encode_views(images)
        view_features = view_features + self._positional_embedding(view_features, poses)
        fused = self.fuse_features(view_features)
        return self.head(fused)
