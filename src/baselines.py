from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _dft_matrix(n: int, device: torch.device | None = None) -> Tensor:
    idx = torch.arange(n, dtype=torch.float32, device=device)
    angle = -2.0 * math.pi * idx.view(n, 1) * idx.view(1, n) / n
    return torch.complex(torch.cos(angle), torch.sin(angle)).to(torch.complex64)


class FixedDFTOracle(nn.Module):
    """Non-learned oracle using the DFT matrix for comparison."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.register_buffer("A", _dft_matrix(n))
        self.register_buffer("B", _dft_matrix(n))

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x_complex = x.to(device=self.A.device, dtype=torch.complex64)
        h_complex = h.to(device=self.A.device, dtype=torch.complex64)
        x_features = x_complex @ self.A.T
        h_features = h_complex @ self.B.T
        product = x_features * h_features
        return product @ torch.linalg.inv(self.A).T


class MLPBaseline(nn.Module):
    """Real-valued baseline mapping concatenated inputs to the convolution output."""

    def __init__(self, n: int, hidden: int = 256, depth: int = 3) -> None:
        super().__init__()
        self.n = n
        layers: list[nn.Module] = []
        in_dim = 2 * n
        for layer_idx in range(depth):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, n))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return self.net(torch.cat([x.float(), h.float()], dim=-1))


class CNNBaseline(nn.Module):
    """Small 1-D convolutional baseline.

    This explicitly-labeled baseline uses ordinary Conv1d layers for comparison.
    The direct-data and learned spectral model modules do not use built-in
    convolution.
    """

    def __init__(self, n: int, channels: int = 64, depth: int = 4, kernel_size: int = 5) -> None:
        super().__init__()
        self.n = n
        padding = kernel_size // 2
        layers: list[nn.Module] = [nn.Conv1d(2, channels, kernel_size, padding=padding), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Conv1d(channels, channels, kernel_size, padding=padding), nn.GELU()])
        layers.append(nn.Conv1d(channels, 1, kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        stacked = torch.stack([x.float(), h.float()], dim=1)
        return self.net(stacked).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
