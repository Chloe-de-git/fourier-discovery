from __future__ import annotations

import torch
from torch import Tensor

from src.convolution import circular_conv


Dataset = dict[str, Tensor]


def _generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def _make_sparse_kernels(num: int, n: int, seed: int) -> Tensor:
    gen = _generator(seed)
    h = torch.zeros(num, n, dtype=torch.float32)
    k = max(2, n // 8)
    for row in range(num):
        idx = torch.randperm(n, generator=gen)[:k]
        h[row, idx] = torch.randn(k, generator=gen, dtype=torch.float32)
    return h


def _make_structured_kernels(num: int, n: int, seed: int) -> Tensor:
    gen = _generator(seed)
    noise = torch.randn(num, n, generator=gen, dtype=torch.float32)
    positions = torch.arange(n, dtype=torch.float32).view(1, n)
    rates = torch.empty(num, 1, dtype=torch.float32).uniform_(1.0, 4.0, generator=gen)
    envelope = torch.exp(-rates * positions / max(n - 1, 1))
    return noise * envelope


def make_dataset(n: int, num: int, kernel_dist: str = "dense", seed: int = 0) -> Dataset:
    """Build deterministic random triples ``(x, h, y)`` of length ``n``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if num <= 0:
        raise ValueError("num must be positive")

    gen = _generator(seed)
    x = torch.randn(num, n, generator=gen, dtype=torch.float32)

    if kernel_dist == "dense":
        h = torch.randn(num, n, generator=gen, dtype=torch.float32)
    elif kernel_dist == "sparse":
        h = _make_sparse_kernels(num, n, seed + 10_000)
    elif kernel_dist == "structured":
        h = _make_structured_kernels(num, n, seed + 20_000)
    else:
        raise ValueError(f"unknown kernel_dist {kernel_dist!r}")

    y = circular_conv(x, h).to(torch.float32)
    return {"x": x, "h": h, "y": y}


def standard_splits(
    n: int,
    train_num: int = 10_000,
    test_num: int = 2_000,
    ood_num: int = 2_000,
    seed: int = 0,
) -> dict[str, Dataset]:
    """Return dense train/test splits plus sparse and structured OOD splits."""
    return {
        "train": make_dataset(n, train_num, "dense", seed),
        "test": make_dataset(n, test_num, "dense", seed + 1),
        "ood_sparse": make_dataset(n, ood_num, "sparse", seed + 2),
        "ood_structured": make_dataset(n, ood_num, "structured", seed + 3),
    }
