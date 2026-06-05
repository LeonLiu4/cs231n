"""Point cloud decoders."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """Center and scale to unit sphere, matching normalized ShapeNet GT."""
    points = points - points.mean(dim=1, keepdim=True)
    radius = points.norm(dim=-1).amax(dim=1, keepdim=True).clamp(min=1e-6)
    return points / radius.unsqueeze(-1)


class PointCloudHead(nn.Module):
    """Flat MLP baseline decoder."""

    def __init__(
        self,
        input_dim: int,
        num_points: int = 2048,
        hidden_dim: int = 512,
        output_tanh: bool = True,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.output_tanh = output_tanh
        self.normalize_output = normalize_output
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_points * 3),
        )

    def _finalize_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.normalize_output:
            points = normalize_point_cloud(points)
        elif self.output_tanh:
            points = torch.tanh(points)
        return points

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        points = self.net(features).view(-1, self.num_points, 3)
        return self._finalize_points(points)


class FoldingPointCloudHead(nn.Module):
    """
    FoldingNet-style decoder: global feature + 2D grid -> 3D points.
    Better inductive bias than a single linear map to 2048x3.
    """

    def __init__(
        self,
        input_dim: int,
        num_points: int = 2048,
        hidden_dim: int = 512,
        output_tanh: bool = True,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.output_tanh = output_tanh
        self.normalize_output = normalize_output
        grid_side = int(math.ceil(math.sqrt(num_points)))
        self.grid_side = grid_side
        self.grid_count = grid_side * grid_side

        coords = torch.linspace(-0.5, 0.5, grid_side)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        self.register_buffer("grid", grid)

        self.feature_norm = nn.LayerNorm(input_dim)

        self.fold1 = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.fold2 = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.feature_norm(features)
        batch_size = features.shape[0]
        grid = self.grid.unsqueeze(0).expand(batch_size, -1, -1)
        global_feat = features.unsqueeze(1).expand(-1, self.grid_count, -1)

        fold1_in = torch.cat([global_feat, grid], dim=-1)
        coarse = self.fold1(fold1_in)

        fold2_in = torch.cat([global_feat, coarse], dim=-1)
        points = self.fold2(fold2_in)

        if self.grid_count > self.num_points:
            points = points[:, : self.num_points]
        elif self.grid_count < self.num_points:
            pad = points[:, -1:].expand(-1, self.num_points - self.grid_count, -1)
            points = torch.cat([points, pad], dim=1)
        return self._finalize_points(points)

    def _finalize_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.normalize_output:
            points = normalize_point_cloud(points)
        elif self.output_tanh:
            points = torch.tanh(points)
        return points


def build_point_cloud_head(
    head_type: str,
    input_dim: int,
    num_points: int,
    hidden_dim: int,
    output_tanh: bool = True,
    normalize_output: bool = False,
) -> nn.Module:
    if head_type == "mlp":
        return PointCloudHead(
            input_dim,
            num_points,
            hidden_dim,
            output_tanh=output_tanh,
            normalize_output=normalize_output,
        )
    if head_type == "folding":
        return FoldingPointCloudHead(
            input_dim,
            num_points,
            hidden_dim,
            output_tanh=output_tanh,
            normalize_output=normalize_output,
        )
    raise ValueError(f"Unknown head_type: {head_type}")
