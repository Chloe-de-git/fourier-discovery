from __future__ import annotations

import torch
from torch import Tensor, nn


def _complex_eye_plus_noise(n: int, scale: float = 0.01, seed: int | None = None) -> Tensor:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    real = torch.randn(n, n, generator=gen, dtype=torch.float32)
    imag = torch.randn(n, n, generator=gen, dtype=torch.float32)
    noise = torch.complex(real, imag).to(torch.complex64) * scale
    return torch.eye(n, dtype=torch.complex64) + noise


class SpectralConvUntied(nn.Module):
    """Unconstrained complex matrices A and B for learned convolution."""

    def __init__(self, n: int, init_scale: float = 0.01, seed: int | None = None) -> None:
        super().__init__()
        self.n = n
        self.A = nn.Parameter(_complex_eye_plus_noise(n, init_scale, seed))
        self.B = nn.Parameter(_complex_eye_plus_noise(n, init_scale, None if seed is None else seed + 1))

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x_complex = x.to(dtype=torch.complex64)
        h_complex = h.to(dtype=torch.complex64)
        x_features = x_complex @ self.A.T
        h_features = h_complex @ self.B.T
        product = x_features * h_features
        return product @ torch.linalg.inv(self.A).T


class SpectralConvTied(nn.Module):
    """Unconstrained complex matrix A with B tied to A."""

    def __init__(self, n: int, init_scale: float = 0.01, seed: int | None = None) -> None:
        super().__init__()
        self.n = n
        self.A = nn.Parameter(_complex_eye_plus_noise(n, init_scale, seed))

    @property
    def B(self) -> Tensor:
        return self.A

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x_complex = x.to(dtype=torch.complex64)
        h_complex = h.to(dtype=torch.complex64)
        x_features = x_complex @ self.A.T
        h_features = h_complex @ self.A.T
        product = x_features * h_features
        return product @ torch.linalg.inv(self.A).T
