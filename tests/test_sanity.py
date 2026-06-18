from __future__ import annotations

from pathlib import Path

import torch

from analysis.metrics import mean_diagonalization_residual
from src.baselines import HardProductLearnedBasis, HardProductOracle
from src.convolution import circular_conv, circulant_matrix
from src.data import make_dataset
from src.models import SpectralNeuralModel
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


def test_hard_product_oracle_reproduces_convolution() -> None:
    ds = make_dataset(16, 32, seed=2)
    oracle = HardProductOracle(16)
    pred = oracle(ds["x"], ds["h"])
    loss = complex_mse(pred, ds["y"])
    assert float(loss) < 1e-8


def test_hard_product_learned_basis_short_training() -> None:
    train_ds = make_dataset(16, 512, seed=3)
    test_ds = make_dataset(16, 128, seed=4)
    model = HardProductLearnedBasis(16, seed=5)
    history = train(
        model,
        train_ds,
        test_ds,
        steps=700,
        lr=1e-2,
        batch=128,
        log_every=250,
        seed=6,
        save=False,
    )
    assert history["final_test_mse"] < 1e-3


def test_spectral_neural_model_short_training_pipeline() -> None:
    train_ds = make_dataset(16, 512, seed=7)
    test_ds = make_dataset(16, 128, seed=8)
    model = SpectralNeuralModel(16, hidden=64, seed=4)
    train(
        model,
        train_ds,
        test_ds,
        steps=1_500,
        lr=1e-3,
        batch=128,
        log_every=1_500,
        seed=10,
        save=False,
        restore_best=True,
        diag_loss_weight=20.0,
        diag_loss_kernels=16,
        gauge_row_norm=1.0,
    )
    model.A.requires_grad_(False)
    train(
        model,
        train_ds,
        test_ds,
        steps=3_000,
        lr=1e-3,
        batch=128,
        log_every=3_000,
        seed=1_010,
        save=False,
        restore_best=True,
        encoded_loss_weight=1.0,
        gauge_row_norm=1.0,
    )
    history = train(
        model,
        train_ds,
        test_ds,
        steps=1_000,
        lr=3e-4,
        batch=128,
        log_every=1_000,
        seed=2_010,
        save=False,
        restore_best=True,
        encoded_loss_weight=1.0,
        gauge_row_norm=1.0,
    )
    residual = mean_diagonalization_residual(model.A.detach().cpu(), test_ds["h"][:16])
    assert history["final_test_mse"] < 1e-2
    assert residual < 0.05


def test_constrained_files_do_not_use_forbidden_apis() -> None:
    root = Path(__file__).resolve().parents[1]
    constrained = [
        root / "src" / "data.py",
        root / "src" / "convolution.py",
        root / "src" / "models.py",
        root / "src" / "train.py",
    ]
    forbidden = [
        "torch.fft",
        "numpy.fft",
        "np.fft",
        "scipy.fft",
        "scipy.fftpack",
        "conv1d",
        "conv2d",
        "numpy.convolve",
        "np.convolve",
        "scipy.signal",
        "dft_matrix",
        "_dft",
    ]
    for path in constrained:
        text = path.read_text(encoding="utf-8").lower()
        for needle in forbidden:
            assert needle not in text, f"{needle!r} found in {path}"
