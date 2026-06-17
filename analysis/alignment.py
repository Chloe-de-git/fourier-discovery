from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def row_normalize(matrix: Tensor, eps: float = 1e-12) -> Tensor:
    norms = torch.linalg.norm(matrix, dim=1, keepdim=True).clamp_min(eps)
    return matrix / norms


def row_assignment(source: Tensor, target: Tensor) -> tuple[np.ndarray, np.ndarray, Tensor]:
    """Match rows by maximum absolute complex correlation."""
    source_hat = row_normalize(source)
    target_hat = row_normalize(target)
    corr = source_hat.conj() @ target_hat.T
    cost = -corr.abs().detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind, corr


def optimal_complex_scale(source_row: Tensor, target_row: Tensor, eps: float = 1e-12) -> Tensor:
    denom = torch.vdot(source_row, source_row)
    if denom.abs() < eps:
        return torch.zeros((), dtype=source_row.dtype, device=source_row.device)
    return torch.vdot(source_row, target_row) / denom


def scaled_row_residual(source_row: Tensor, target_row: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    scale = optimal_complex_scale(source_row, target_row, eps)
    residual = torch.linalg.norm(scale * source_row - target_row) / torch.linalg.norm(target_row).clamp_min(eps)
    return residual, scale
