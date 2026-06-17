from __future__ import annotations

import torch

from analysis.metrics import mean_diagonalization_residual
from src.baselines import FixedDFTOracle
from src.convolution import circular_conv, circulant_matrix
from src.data import make_dataset
from src.models import SpectralConvUntied
from src.train import complex_mse, train


def test_circulant_matrix_matches_circular_conv() -> None:
    torch.manual_seed(0)
    n = 12
    x = torch.randn(n)
    h = torch.randn(n)
    y_matrix = circulant_matrix(h) @ x
    y_direct = circular_conv(x, h)[0]
    assert torch.allclose(y_matrix, y_direct, atol=1e-5, rtol=1e-5)


def test_circular_conv_commutative() -> None:
    torch.manual_seed(1)
    x = torch.randn(5, 16)
    h = torch.randn(5, 16)
    assert torch.allclose(circular_conv(x, h), circular_conv(h, x), atol=1e-5, rtol=1e-5)


def test_fixed_dft_oracle_reproduces_convolution() -> None:
    ds = make_dataset(16, 32, seed=2)
    oracle = FixedDFTOracle(16)
    pred = oracle(ds["x"], ds["h"])
    loss = complex_mse(pred, ds["y"])
    assert float(loss) < 1e-8


def test_short_untied_training_learns_diagonalizer() -> None:
    train_ds = make_dataset(16, 512, seed=3)
    test_ds = make_dataset(16, 128, seed=4)
    model = SpectralConvUntied(16, seed=5)
    history = train(
        model,
        train_ds,
        test_ds,
        steps=500,
        lr=1e-2,
        batch=128,
        log_every=250,
        run_name="test_short_untied",
        seed=6,
    )
    residual = mean_diagonalization_residual(model.A.detach().cpu(), test_ds["h"][:16])
    assert history["final_test_mse"] < 1e-3
    assert residual < 1e-2
