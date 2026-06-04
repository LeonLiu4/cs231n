"""Chamfer distance loss."""

from __future__ import annotations

import torch


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred, target: [B, N, 3]
    Returns (loss, pred_to_target, target_to_pred) mean distances.
    """
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = torch.sum(diff * diff, dim=-1)

    pred_to_target = dist.min(dim=2).values.mean()
    target_to_pred = dist.min(dim=1).values.mean()
    loss = pred_to_target + target_to_pred
    return loss, pred_to_target, target_to_pred
