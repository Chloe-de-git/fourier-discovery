from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from analysis.alignment import row_assignment, scaled_row_residual
from src.convolution import circulant_matrix


def dft_matrix(n: int, device: torch.device | None = None) -> Tensor:
    idx = torch.arange(n, dtype=torch.float32, device=device)
    angle = -2.0 * math.pi * idx.view(n, 1) * idx.view(1, n) / n
    return torch.complex(torch.cos(angle), torch.sin(angle)).to(torch.complex64)


def diagonalization_residual(A: Tensor, h: Tensor) -> float:
    A_detached = A.detach().to(torch.complex64)
    h_real = h.detach().to(device=A_detached.device).real.float()
    C = circulant_matrix(h_real).to(torch.complex64)
    M = A_detached @ C @ torch.linalg.inv(A_detached)
    total = (M.abs() ** 2).sum()
    if float(total.detach().cpu()) == 0.0:
        return 0.0
    diag = torch.diag(torch.diagonal(M))
    offdiag = M - diag
    return float(((offdiag.abs() ** 2).sum() / total).detach().cpu())


def mean_diagonalization_residual(A: Tensor, kernels: Tensor) -> float:
    kernels_batched = kernels.unsqueeze(0) if kernels.ndim == 1 else kernels
    values = [diagonalization_residual(A, h) for h in kernels_batched]
    return float(sum(values) / len(values))


def dft_alignment_error(A: Tensor) -> dict[str, Any]:
    A_detached = A.detach().to(torch.complex64)
    F = dft_matrix(A_detached.shape[0], device=A_detached.device)
    row_ind, col_ind, corr = row_assignment(A_detached, F)

    perm = [-1] * A_detached.shape[0]
    residuals: list[float] = []
    abs_corrs: list[float] = []
    for row, col in zip(row_ind, col_ind, strict=True):
        row_i = int(row)
        col_i = int(col)
        residual, _ = scaled_row_residual(A_detached[row_i], F[col_i])
        perm[row_i] = col_i
        residuals.append(float(residual.detach().cpu()))
        abs_corrs.append(float(corr[row_i, col_i].abs().detach().cpu()))

    return {
        "perm": perm,
        "mean_row_residual": float(sum(residuals) / len(residuals)),
        "mean_abs_corr": float(sum(abs_corrs) / len(abs_corrs)),
        "row_residuals": residuals,
    }


def spectrum_match(model: Any, h: Tensor) -> dict[str, float]:
    A = getattr(model, "A", None)
    if A is None:
        raise ValueError("model must expose A")
    A_detached = A.detach().to(torch.complex64)
    device = A_detached.device
    h_complex = h.detach().to(device=device, dtype=torch.complex64)
    F = dft_matrix(h_complex.shape[0], device=device)
    true_spectrum = F @ h_complex

    if hasattr(model, "B"):
        B = getattr(model, "B").detach().to(device=device, dtype=torch.complex64)
        learned = B @ h_complex
    else:
        C = circulant_matrix(h_complex.real.float()).to(device=device, dtype=torch.complex64)
        learned = torch.diagonal(A_detached @ C @ torch.linalg.inv(A_detached))

    alignment = dft_alignment_error(A_detached)
    perm = alignment["perm"]
    if all(col >= 0 for col in perm):
        ordered_true = true_spectrum[torch.tensor(perm, device=device)]
        rel = torch.linalg.norm(learned - ordered_true) / torch.linalg.norm(ordered_true).clamp_min(1e-12)
        return {"relative_error": float(rel.detach().cpu())}

    learned_sorted = torch.sort(learned.abs()).values
    true_sorted = torch.sort(true_spectrum.abs()).values
    rel = torch.linalg.norm(learned_sorted - true_sorted) / torch.linalg.norm(true_sorted).clamp_min(1e-12)
    return {"relative_error": float(rel.detach().cpu())}


def unitarity_stats(A: Tensor) -> dict[str, float]:
    A_detached = A.detach().to(torch.complex64)
    abs_values = A_detached.abs()
    gram = A_detached.conj().T @ A_detached
    n = A_detached.shape[0]
    c = torch.trace(gram) / n
    target = c * torch.eye(n, dtype=torch.complex64, device=A_detached.device)
    scaled_error = torch.linalg.norm(gram - target) / torch.linalg.norm(target).clamp_min(1e-12)
    abs_mean = abs_values.mean()
    abs_std = abs_values.std(unbiased=False)
    return {
        "abs_mean": float(abs_mean.detach().cpu()),
        "abs_std_over_mean": float((abs_std / abs_mean.clamp_min(1e-12)).detach().cpu()),
        "scaled_unitarity_error": float(scaled_error.detach().cpu()),
    }
