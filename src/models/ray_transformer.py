"""Ray-aware patch-token reconstruction model.

Per-view DINOv2 patch tokens are augmented with an embedding that grounds each
token in 3D, fused across views by a Transformer, and decoded to a 2048x3 point
cloud by a set of learned point queries. The variants differ only in the
per-patch embedding added to the tokens:

    pos_embedding="2d_positional"  -> learned 2-D patch-grid code only
    pos_embedding="view_id"        -> + learned per-view-index embedding
    pos_embedding="camera_pose"    -> + MLP over the 7-D camera pose vector
    pos_embedding="geometry_aware" -> + per-patch camera-ray embedding

The renders are orthographic, so a view's rays are parallel: the per-patch
signal is the ray origin on the image plane (the 3-D point each patch samples
through), while the viewing direction is shared across the view.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .dinov2_encoder import DINOv2Encoder
from .pose_aware_baseline import POSE_VECTOR_DIM, _MLP, _normalize_pose

# Orthographic image-plane scale used by src/data/render.py.
ORTHO_SCALE_FACTOR = 0.42


def _camera_basis(
    poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (eye, right, up, forward) world-space basis from pose vectors.

    Mirrors ``CameraPose.view_matrix``: look at the origin with world up +z,
    falling back to +y world up when looking straight down the axis.
    """
    eye = poses[..., 3:6]
    forward = -eye / (eye.norm(dim=-1, keepdim=True) + 1e-8)
    world_up = torch.zeros_like(forward)
    world_up[..., 2] = 1.0
    right = torch.cross(forward, world_up, dim=-1)
    degenerate = right.norm(dim=-1, keepdim=True) < 1e-6
    alt_up = torch.zeros_like(forward)
    alt_up[..., 1] = 1.0
    right = torch.where(degenerate, torch.cross(forward, alt_up, dim=-1), right)
    right = right / (right.norm(dim=-1, keepdim=True) + 1e-8)
    up = torch.cross(right, forward, dim=-1)
    up = up / (up.norm(dim=-1, keepdim=True) + 1e-8)
    return eye, right, up, forward


def _fourier(coords: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Sin/cos Fourier features of 3-D coords: ``[..., 3] -> [..., 3 * 2 * F]``."""
    freqs = 2.0 ** torch.arange(num_freqs, device=coords.device, dtype=coords.dtype)
    scaled = coords.unsqueeze(-1) * (freqs * math.pi)
    feats = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
    return feats.flatten(-2)


class RayTransformerReconstructor(nn.Module):
    """Patch-token Transformer reconstructor with per-patch 3-D embeddings.

    The frozen DINOv2 backbone produces a patch-token sequence per view; an
    embedding (selected by ``pos_embedding``) is added to every token, all
    tokens are fused by a Transformer encoder, and learned point queries
    cross-attend to the fused tokens to emit the point cloud.
    """

    def __init__(
        self,
        *,
        dinov2_variant: str = "dinov2_vitb14",
        num_output_points: int = 2048,
        hidden_dim: int = 512,
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
        num_views: int = 4,
        pos_embedding: str = "geometry_aware",
        image_size: int = 224,
        d_model: int = 384,
        nhead: int = 6,
        enc_layers: int = 4,
        dec_layers: int = 2,
        dim_feedforward: int = 1024,
        num_queries: int = 256,
        geometry_num_freqs: int = 6,
        pose_embed_hidden: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_output_points % num_queries != 0:
            raise ValueError("num_output_points must be divisible by num_queries.")

        self.pos_embedding = pos_embedding
        self.num_views = num_views
        self.image_size = image_size
        self.num_output_points = num_output_points
        self.points_per_query = num_output_points // num_queries
        self.geometry_num_freqs = geometry_num_freqs

        self.encoder = DINOv2Encoder(
            dinov2_variant,
            freeze=freeze_backbone,
            feature_mode="patch_tokens",
            unfreeze_last_blocks=unfreeze_last_blocks,
        )
        embed_dim = self.encoder.out_dim
        self.patch_size = self.encoder.patch_size
        self.grid = image_size // self.patch_size
        self.num_patches = self.grid * self.grid

        self.input_proj = nn.Linear(embed_dim, d_model)
        # Shared 2-D patch-grid code added by every variant.
        self.patch_pos = nn.Parameter(torch.zeros(self.num_patches, d_model))
        nn.init.trunc_normal_(self.patch_pos, std=0.02)

        if pos_embedding == "2d_positional":
            pass
        elif pos_embedding == "view_id":
            self.view_embedding = nn.Embedding(num_views, d_model)
        elif pos_embedding == "camera_pose":
            self.pose_encoder = _MLP(POSE_VECTOR_DIM, pose_embed_hidden, d_model)
        elif pos_embedding == "geometry_aware":
            ray_dim = 3 * 2 * geometry_num_freqs + 3 + 1  # fourier(origin) + dir + dist
            self.ray_encoder = _MLP(ray_dim, pose_embed_hidden, d_model)
        else:
            raise ValueError(
                f"Unknown pos_embedding: {pos_embedding!r}. Use one of "
                "'2d_positional', 'view_id', 'camera_pose', 'geometry_aware'."
            )

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(enc_layer, enc_layers)

        self.query_embedding = nn.Parameter(torch.zeros(num_queries, d_model))
        nn.init.trunc_normal_(self.query_embedding, std=0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.points_per_query * 3),
        )

    def _patch_centers(self, device, dtype) -> torch.Tensor:
        """Pixel coordinates of each patch center as ``[num_patches, 2]`` (px, py)."""
        coords = (torch.arange(self.grid, device=device, dtype=dtype) + 0.5) * self.patch_size
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    def _ray_features(self, poses: torch.Tensor) -> torch.Tensor:
        """Per-patch ray embedding input from poses ``[B, V, 7] -> [B, V, P, *]``."""
        batch_size, num_views = poses.shape[:2]
        eye, right, up, forward = _camera_basis(poses)

        centers = self._patch_centers(poses.device, poses.dtype)
        center = self.image_size / 2.0
        scale = self.image_size * ORTHO_SCALE_FACTOR
        x_cam = (centers[:, 0] - center) / scale
        y_cam = (center - centers[:, 1]) / scale

        origin = (
            eye[:, :, None, :]
            + x_cam[None, None, :, None] * right[:, :, None, :]
            + y_cam[None, None, :, None] * up[:, :, None, :]
        )
        fourier = _fourier(origin, self.geometry_num_freqs)
        direction = forward[:, :, None, :].expand(batch_size, num_views, self.num_patches, 3)
        distance = poses[..., 2:3][:, :, None, :].expand(
            batch_size, num_views, self.num_patches, 1
        ) / 2.0
        return torch.cat([fourier, direction, distance], dim=-1)

    def _patch_embedding(self, tokens: torch.Tensor, poses: torch.Tensor | None) -> torch.Tensor:
        batch_size, num_views = tokens.shape[:2]
        emb = self.patch_pos[None, None].expand(batch_size, num_views, -1, -1)

        if self.pos_embedding == "view_id":
            idx = torch.arange(num_views, device=tokens.device)
            return emb + self.view_embedding(idx)[None, :, None, :]

        if self.pos_embedding in ("camera_pose", "geometry_aware") and poses is None:
            raise ValueError(f"pos_embedding={self.pos_embedding!r} requires poses, got None.")

        if self.pos_embedding == "camera_pose":
            return emb + self.pose_encoder(_normalize_pose(poses))[:, :, None, :]
        if self.pos_embedding == "geometry_aware":
            return emb + self.ray_encoder(self._ray_features(poses))
        return emb

    def forward(self, images: torch.Tensor, poses: torch.Tensor | None = None) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if poses is not None and poses.ndim == 2:
            poses = poses.unsqueeze(1)
        batch_size, num_views = images.shape[:2]

        flat = images.reshape(batch_size * num_views, *images.shape[2:])
        tokens = self.input_proj(self.encoder(flat))
        tokens = tokens.view(batch_size, num_views, self.num_patches, -1)
        tokens = tokens + self._patch_embedding(tokens, poses)

        memory = self.fusion(tokens.reshape(batch_size, num_views * self.num_patches, -1))
        queries = self.query_embedding[None].expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        points = self.head(decoded)
        return points.reshape(batch_size, self.num_output_points, 3)
