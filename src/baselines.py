from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.models import CoordwiseMLP, _complex_eye_plus_noise


def _dft_matrix(n: int, device: torch.device | None = None) -> Tensor:
    idx = torch.arange(n, dtype=torch.float32, device=device)
    angle = -2.0 * math.pi * idx.view(n, 1) * idx.view(1, n) / n
    return torch.complex(torch.cos(angle), torch.sin(angle)).to(torch.complex64)


class HardProductOracle(nn.Module):
    """DFT basis plus hard-coded elementwise product, for oracle comparison only."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.register_buffer("A", _dft_matrix(n))
        self.register_buffer("B", _dft_matrix(n))

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x_complex = x.to(device=self.A.device, dtype=torch.complex64)
        h_complex = h.to(device=self.A.device, dtype=torch.complex64)
        u = x_complex @ self.A.T
        v = h_complex @ self.B.T
        return torch.linalg.solve(self.A, (u * v).T).T


class HardProductLearnedBasis(nn.Module):
    """Learnable unconstrained complex basis with a hard-coded coordinate product."""

    def __init__(self, n: int, init_scale: float = 0.01, seed: int | None = None) -> None:
        super().__init__()
        self.n = n
        self.A = nn.Parameter(_complex_eye_plus_noise(n, init_scale, seed))

    @property
    def B(self) -> Tensor:
        return self.A

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x_complex = x.to(device=self.A.device, dtype=torch.complex64)
        h_complex = h.to(device=self.A.device, dtype=torch.complex64)
        u = x_complex @ self.A.T
        v = h_complex @ self.A.T
        return torch.linalg.solve(self.A, (u * v).T).T


class BilinearTensor(nn.Module):
    """Fully general bilinear map y[t] = sum_ij W[t,i,j] x[i] h[j]."""

    def __init__(self, n: int, init_scale: float = 0.01, seed: int | None = None) -> None:
        super().__init__()
        self.n = n
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        real = torch.randn(n, n, n, generator=gen, dtype=torch.float32) * init_scale
        imag = torch.randn(n, n, n, generator=gen, dtype=torch.float32) * init_scale
        self.W = nn.Parameter(torch.complex(real, imag).to(torch.complex64))

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x_complex = x.to(device=self.W.device, dtype=torch.complex64)
        h_complex = h.to(device=self.W.device, dtype=torch.complex64)
        return torch.einsum("bi,bj,tij->bt", x_complex, h_complex, self.W)


class MLPControl(nn.Module):
    """Real-valued MLP mapping concatenated inputs directly to the convolution output."""

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


class NonlinearEncoderControl(nn.Module):
    """Nonlinear encoders feeding the same coordinate-wise interaction idea."""

    def __init__(
        self,
        n: int,
        hidden: int = 128,
        coord_hidden: int = 64,
        coord_depth: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.x_encoder = nn.Sequential(nn.Linear(n, hidden), nn.GELU(), nn.Linear(hidden, 2 * n))
        self.h_encoder = nn.Sequential(nn.Linear(n, hidden), nn.GELU(), nn.Linear(hidden, 2 * n))
        self.g_phi = CoordwiseMLP(hidden=coord_hidden, depth=coord_depth, activation="tanh")
        self.D = nn.Parameter(_complex_eye_plus_noise(n, 0.01, seed))

    def encode(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        u_raw = self.x_encoder(x.float()).view(x.shape[0], self.n, 2)
        v_raw = self.h_encoder(h.float()).view(h.shape[0], self.n, 2)
        u = torch.complex(u_raw[..., 0], u_raw[..., 1]).to(torch.complex64)
        v = torch.complex(v_raw[..., 0], v_raw[..., 1]).to(torch.complex64)
        return u, v

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        u, v = self.encode(x, h)
        u_raw = torch.stack([u.real, u.imag], dim=-1)
        v_raw = torch.stack([v.real, v.imag], dim=-1)
        features = torch.stack([u_raw[..., 0], u_raw[..., 1], v_raw[..., 0], v_raw[..., 1]], dim=-1)
        out = self.g_phi(features)
        z = torch.complex(out[..., 0], out[..., 1]).to(torch.complex64)
        return z @ self.D.T


class CNNBaseline(nn.Module):
    """Small real Conv1d control retained for backward compatibility."""

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


FixedDFTOracle = HardProductOracle
MLPBaseline = MLPControl


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
