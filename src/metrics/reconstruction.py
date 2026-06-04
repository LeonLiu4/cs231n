"""Reconstruction evaluation metrics."""

from __future__ import annotations

import torch


@torch.no_grad()
def f_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.01,
) -> torch.Tensor:
    """
    F-score at threshold relative to bounding box diagonal.
    pred, target: [B, N, 3]
    """
    combined = torch.cat([pred, target], dim=1)
    bbox_min = combined.min(dim=1).values
    bbox_max = combined.max(dim=1).values
    diag = torch.linalg.norm(bbox_max - bbox_min, dim=-1).clamp_min(1e-8)
    thresh = threshold * diag

    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-8)

    precision = (dist.min(dim=2).values < thresh.unsqueeze(1)).float().mean(dim=1)
    recall = (dist.min(dim=1).values < thresh.unsqueeze(1)).float().mean(dim=1)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean()
