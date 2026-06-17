from __future__ import annotations

import torch
from torch import Tensor


def _as_batched(signal: Tensor, name: str) -> Tensor:
    if signal.ndim == 1:
        return signal.unsqueeze(0)
    if signal.ndim == 2:
        return signal
    raise ValueError(f"{name} must have shape (n,) or (batch, n); got {tuple(signal.shape)}")


def circular_conv(x: Tensor, h: Tensor) -> Tensor:
    """Circular convolution from the defining sum.

    Inputs may be single vectors of shape ``(n,)`` or batches of shape
    ``(batch, n)``. Single vectors are treated as a batch of one, so the return
    value always has shape ``(batch, n)``.
    """
    x_b = _as_batched(x, "x")
    h_b = _as_batched(h, "h")

    if x_b.shape[-1] != h_b.shape[-1]:
        raise ValueError(f"x and h must have the same length; got {x_b.shape[-1]} and {h_b.shape[-1]}")
    if x_b.shape[0] != h_b.shape[0]:
        if x_b.shape[0] == 1:
            x_b = x_b.expand(h_b.shape[0], -1)
        elif h_b.shape[0] == 1:
            h_b = h_b.expand(x_b.shape[0], -1)
        else:
            raise ValueError(f"batch sizes must match or be one; got {x_b.shape[0]} and {h_b.shape[0]}")

    n = x_b.shape[-1]
    y = torch.zeros(x_b.shape, dtype=torch.promote_types(x_b.dtype, h_b.dtype), device=x_b.device)
    for s in range(n):
        y = y + x_b[:, s : s + 1] * torch.roll(h_b, shifts=s, dims=1)
    return y


def circulant_matrix(h: Tensor) -> Tensor:
    """Return C with C[t, s] = h[(t - s) mod n]."""
    if h.ndim != 1:
        raise ValueError(f"h must have shape (n,); got {tuple(h.shape)}")
    n = h.shape[0]
    rows = torch.arange(n, device=h.device).view(n, 1)
    cols = torch.arange(n, device=h.device).view(1, n)
    indices = (rows - cols) % n
    return h[indices]
