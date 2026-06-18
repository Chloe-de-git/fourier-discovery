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


def _model_device(model: Any) -> torch.device:
    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.device
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device
    return torch.device("cpu")


@torch.no_grad()
def _evaluate_g_phi(model: Any, u: Tensor, v: Tensor) -> Tensor:
    if not hasattr(model, "g_phi"):
        raise ValueError("model must expose g_phi")
    features = torch.stack([u.real, u.imag, v.real, v.imag], dim=-1)
    out = model.g_phi(features)
    return torch.complex(out[..., 0], out[..., 1]).to(torch.complex64)


@torch.no_grad()
def multiplication_probe(
    model: Any,
    data: dict[str, Tensor] | None = None,
    mag_range: float | tuple[float, float] | None = None,
    num: int = 8_192,
    seed: int = 0,
    mode: str = "encoded",
) -> dict[str, Any]:
    """Probe whether g_phi matches complex multiplication up to one scale."""
    if num <= 0:
        raise ValueError("num must be positive")

    device = _model_device(model)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if data is not None and mode == "encoded":
        if hasattr(model, "encode"):
            u, v = model.encode(data["x"].to(device), data["h"].to(device))
        else:
            A = getattr(model, "A")
            B = getattr(model, "B", A)
            x_complex = data["x"].to(device=device, dtype=torch.complex64)
            h_complex = data["h"].to(device=device, dtype=torch.complex64)
            u = x_complex @ A.T
            v = h_complex @ B.T
        u_flat = u.reshape(-1)
        v_flat = v.reshape(-1)
        count = min(num, u_flat.numel())
        idx = torch.randperm(u_flat.numel(), generator=gen, device=device)[:count]
        u_probe = u_flat[idx]
        v_probe = v_flat[idx]
    else:
        if mag_range is None and data is not None:
            if hasattr(model, "encode"):
                u_data, v_data = model.encode(data["x"].to(device), data["h"].to(device))
            else:
                A = getattr(model, "A")
                B = getattr(model, "B", A)
                u_data = data["x"].to(device=device, dtype=torch.complex64) @ A.T
                v_data = data["h"].to(device=device, dtype=torch.complex64) @ B.T
            max_mag = torch.quantile(torch.cat([u_data.abs().flatten(), v_data.abs().flatten()]), 0.95)
            lo, hi = 0.0, float(max_mag.detach().cpu())
        elif isinstance(mag_range, tuple):
            lo, hi = float(mag_range[0]), float(mag_range[1])
        elif mag_range is None:
            lo, hi = 0.0, 2.0
        else:
            lo, hi = 0.0, float(mag_range)

        radius_u = lo + (hi - lo) * torch.rand(num, generator=gen, device=device)
        radius_v = lo + (hi - lo) * torch.rand(num, generator=gen, device=device)
        phase_u = 2.0 * math.pi * torch.rand(num, generator=gen, device=device)
        phase_v = 2.0 * math.pi * torch.rand(num, generator=gen, device=device)
        u_probe = torch.complex(radius_u * torch.cos(phase_u), radius_u * torch.sin(phase_u)).to(torch.complex64)
        v_probe = torch.complex(radius_v * torch.cos(phase_v), radius_v * torch.sin(phase_v)).to(torch.complex64)

    g_pred = _evaluate_g_phi(model, u_probe, v_probe).reshape(-1)
    product = (u_probe * v_probe).reshape(-1)
    denom = torch.vdot(product, product)
    if denom.abs() < 1e-12:
        scale = torch.zeros((), dtype=torch.complex64, device=device)
    else:
        scale = torch.vdot(product, g_pred) / denom
    target = scale * product
    rel = torch.linalg.norm(g_pred - target) / torch.linalg.norm(target).clamp_min(1e-12)

    return {
        "mode": mode,
        "num": int(g_pred.numel()),
        "rel_residual": float(rel.detach().cpu()),
        "scale": complex(scale.detach().cpu().item()),
        "scale_real": float(scale.real.detach().cpu()),
        "scale_imag": float(scale.imag.detach().cpu()),
        "scale_abs": float(scale.abs().detach().cpu()),
    }
