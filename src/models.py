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


def _seeded_coordwise_mlp(hidden: int, depth: int, activation: str, seed: int | None) -> "CoordwiseMLP":
    if seed is None:
        return CoordwiseMLP(hidden=hidden, depth=depth, activation=activation)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CoordwiseMLP(hidden=hidden, depth=depth, activation=activation)


class CoordwiseMLP(nn.Module):
    """Shared coordinate-wise map from (Re u, Im u, Re v, Im v) to (Re z, Im z)."""

    def __init__(self, hidden: int = 64, depth: int = 3, activation: str = "tanh") -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        activations: dict[str, type[nn.Module]] = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        if activation not in activations:
            raise ValueError(f"unknown activation {activation!r}")
        act = activations[activation]

        layers: list[nn.Module] = []
        in_dim = 4
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act())
            in_dim = hidden
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != 4:
            raise ValueError(f"features must have last dimension 4; got {tuple(features.shape)}")
        return self.net(features.float())


class SpectralNeuralModel(nn.Module):
    """Learn a complex basis and a shared coordinate-wise interaction MLP."""

    def __init__(
        self,
        n: int,
        hidden: int = 64,
        depth: int = 3,
        activation: str = "tanh",
        init_scale: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.A = nn.Parameter(_complex_eye_plus_noise(n, init_scale, seed))
        self.g_phi = _seeded_coordwise_mlp(hidden, depth, activation, None if seed is None else seed + 2)

    @property
    def B(self) -> Tensor:
        return self.A

    def encode(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        x_complex = x.to(device=self.A.device, dtype=torch.complex64)
        h_complex = h.to(device=self.A.device, dtype=torch.complex64)
        return x_complex @ self.A.T, h_complex @ self.A.T

    def interact(self, u: Tensor, v: Tensor) -> Tensor:
        features = torch.stack([u.real, u.imag, v.real, v.imag], dim=-1)
        out = self.g_phi(features)
        return torch.complex(out[..., 0], out[..., 1]).to(torch.complex64)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        u, v = self.encode(x, h)
        z = self.interact(u, v)
        return torch.linalg.solve(self.A, z.T).T


class SpectralNeuralModelUntied(nn.Module):
    """Untied variant with separate encoders for x and h, sharing one interaction MLP."""

    def __init__(
        self,
        n: int,
        hidden: int = 64,
        depth: int = 3,
        activation: str = "tanh",
        init_scale: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.A = nn.Parameter(_complex_eye_plus_noise(n, init_scale, seed))
        self.B = nn.Parameter(_complex_eye_plus_noise(n, init_scale, None if seed is None else seed + 1))
        self.g_phi = _seeded_coordwise_mlp(hidden, depth, activation, None if seed is None else seed + 2)

    def encode(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        x_complex = x.to(device=self.A.device, dtype=torch.complex64)
        h_complex = h.to(device=self.A.device, dtype=torch.complex64)
        return x_complex @ self.A.T, h_complex @ self.B.T

    def interact(self, u: Tensor, v: Tensor) -> Tensor:
        features = torch.stack([u.real, u.imag, v.real, v.imag], dim=-1)
        out = self.g_phi(features)
        return torch.complex(out[..., 0], out[..., 1]).to(torch.complex64)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        u, v = self.encode(x, h)
        z = self.interact(u, v)
        return torch.linalg.solve(self.A, z.T).T


SpectralConvTied = SpectralNeuralModel
SpectralConvUntied = SpectralNeuralModelUntied
