from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from analysis.metrics import dft_alignment_error, mean_diagonalization_residual, unitarity_stats
from src.baselines import CNNBaseline, FixedDFTOracle, MLPBaseline, count_parameters
from src.data import Dataset, make_dataset, standard_splits
from src.models import SpectralConvTied, SpectralConvUntied
from src.train import complex_mse, evaluate_loss, get_device, train


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_dataset(ds: Dataset) -> Dataset:
    return {key: value.clone() for key, value in ds.items()}


def _noisy_dataset(ds: Dataset, snr_db: float, seed: int) -> Dataset:
    out = _copy_dataset(ds)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rms = out["y"].pow(2).mean().sqrt()
    noise_std = rms / (10.0 ** (snr_db / 20.0))
    out["y"] = out["y"] + noise_std * torch.randn(out["y"].shape, generator=gen, dtype=out["y"].dtype)
    return out


@torch.no_grad()
def _real_mse(model: nn.Module, ds: Dataset, device: torch.device) -> float:
    model.eval()
    x = ds["x"].to(device)
    h = ds["h"].to(device)
    y = ds["y"].to(device)
    pred = model(x, h)
    return float(((pred - y) ** 2).mean().detach().cpu())


def _train_real_model(
    model: nn.Module,
    train_ds: Dataset,
    test_ds: Dataset,
    steps: int,
    lr: float,
    batch: int | None,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    model.to(device)
    moved_train = {key: value.to(device) for key, value in train_ds.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_train = moved_train["x"].shape[0]
    for _ in range(steps):
        if batch is None:
            idx = None
            x, h, y = moved_train["x"], moved_train["h"], moved_train["y"]
        else:
            idx = torch.randint(num_train, (batch,), device=device)
            x, h, y = moved_train["x"][idx], moved_train["h"][idx], moved_train["y"][idx]
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, h)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    return {
        "train_mse": _real_mse(model, train_ds, device),
        "test_mse": _real_mse(model, test_ds, device),
    }


def _spectral_summary(model: nn.Module, test_ds: Dataset, max_kernels: int = 64) -> dict[str, float]:
    A = getattr(model, "A").detach().cpu()
    kernels = test_ds["h"][:max_kernels]
    align = dft_alignment_error(A)
    unitary = unitarity_stats(A)
    return {
        "diag_residual": mean_diagonalization_residual(A, kernels),
        "dft_mean_row_residual": align["mean_row_residual"],
        "dft_mean_abs_corr": align["mean_abs_corr"],
        "abs_mean": unitary["abs_mean"],
        "abs_std_over_mean": unitary["abs_std_over_mean"],
        "scaled_unitarity_error": unitary["scaled_unitarity_error"],
    }


def _row_scaled_relative(A: Tensor, B: Tensor) -> float:
    A_cpu = A.detach().cpu().to(torch.complex64)
    B_cpu = B.detach().cpu().to(torch.complex64)
    residuals: list[Tensor] = []
    for row in range(A_cpu.shape[0]):
        denom = torch.vdot(B_cpu[row], B_cpu[row]).abs().clamp_min(1e-12)
        scale = torch.vdot(B_cpu[row], A_cpu[row]) / denom
        residuals.append(torch.linalg.norm(scale * B_cpu[row] - A_cpu[row]) / torch.linalg.norm(A_cpu[row]).clamp_min(1e-12))
    return float(torch.stack(residuals).mean())


def run_all(
    mode: str,
    results_dir: str | Path = "results",
    device_name: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device_name)

    if mode == "smoke":
        n_values = [16]
        train_num, test_num, ood_num = 512, 128, 128
        steps, baseline_steps = 250, 150
        batch = 128
        noise_snrs = [30.0]
    elif mode == "full":
        n_values = [16, 32, 64]
        train_num, test_num, ood_num = 10_000, 2_000, 2_000
        steps, baseline_steps = 4_000, 4_000
        batch = None
        noise_snrs = [40.0, 30.0, 20.0, 10.0, 0.0]
    else:
        raise ValueError("mode must be 'smoke' or 'full'")

    mse_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    for n in n_values:
        splits = standard_splits(n, train_num, test_num, ood_num, seed=100 + n)
        model = SpectralConvUntied(n, seed=1)
        run_name = f"untied_n{n}" if mode == "full" else f"smoke_untied_n{n}"
        hist = train(
            model,
            splits["train"],
            splits["test"],
            steps=steps,
            lr=1e-2,
            batch=batch,
            log_every=max(1, steps // 10),
            results_dir=out_dir,
            run_name=run_name,
            device=device,
            seed=10 + n,
        )
        test_mse = evaluate_loss(model, splits["test"], device)
        sparse_mse = evaluate_loss(model, splits["ood_sparse"], device)
        structured_mse = evaluate_loss(model, splits["ood_structured"], device)
        mse_rows.append(
            {
                "experiment": "size_sweep",
                "model": "untied",
                "n": n,
                "test_mse": test_mse,
                "ood_sparse_mse": sparse_mse,
                "ood_structured_mse": structured_mse,
                "train_seconds": hist["elapsed_sec"],
            }
        )
        diag_rows.append({"experiment": "size_sweep", "model": "untied", "n": n, **_spectral_summary(model, splits["test"])})

    n = 32 if mode == "full" else 16
    splits = standard_splits(n, train_num, test_num, ood_num, seed=232)

    for snr in noise_snrs:
        noisy_train = _noisy_dataset(splits["train"], snr, seed=int(1_000 + snr))
        model = SpectralConvUntied(n, seed=20)
        train(
            model,
            noisy_train,
            splits["test"],
            steps=steps,
            lr=1e-2,
            batch=batch,
            log_every=max(1, steps // 10),
            results_dir=out_dir,
            run_name=f"noise_snr{int(snr)}_n{n}_{mode}",
            device=device,
            seed=30 + int(snr),
        )
        mse_rows.append(
            {
                "experiment": "noise_sweep",
                "model": "untied",
                "n": n,
                "snr_db": snr,
                "test_mse": evaluate_loss(model, splits["test"], device),
            }
        )
        diag_rows.append(
            {
                "experiment": "noise_sweep",
                "model": "untied",
                "n": n,
                "snr_db": snr,
                **_spectral_summary(model, splits["test"]),
            }
        )

    untied = SpectralConvUntied(n, seed=40)
    tied = SpectralConvTied(n, seed=41)
    train(untied, splits["train"], splits["test"], steps=steps, lr=1e-2, batch=batch, log_every=max(1, steps // 10), results_dir=out_dir, run_name=f"tied_compare_untied_n{n}_{mode}", device=device, seed=40)
    train(tied, splits["train"], splits["test"], steps=steps, lr=1e-2, batch=batch, log_every=max(1, steps // 10), results_dir=out_dir, run_name=f"tied_compare_tied_n{n}_{mode}", device=device, seed=41)
    for name, model in [("untied", untied), ("tied", tied)]:
        row = {
            "experiment": "tied_vs_untied",
            "model": name,
            "n": n,
            "test_mse": evaluate_loss(model, splits["test"], device),
        }
        if name == "untied":
            row["row_scaled_A_B_relative"] = _row_scaled_relative(model.A, model.B)
        mse_rows.append(row)
        diag_rows.append({"experiment": "tied_vs_untied", "model": name, "n": n, **_spectral_summary(model, splits["test"])})

    oracle = FixedDFTOracle(n).to(device)
    mse_rows.append(
        {
            "experiment": "baselines",
            "model": "FixedDFTOracle",
            "n": n,
            "test_mse": evaluate_loss(oracle, splits["test"], device),
            "ood_sparse_mse": evaluate_loss(oracle, splits["ood_sparse"], device),
            "ood_structured_mse": evaluate_loss(oracle, splits["ood_structured"], device),
            "params": count_parameters(oracle),
        }
    )

    for baseline_name, baseline in [("MLPBaseline", MLPBaseline(n)), ("CNNBaseline", CNNBaseline(n))]:
        metrics = _train_real_model(baseline, splits["train"], splits["test"], baseline_steps, 1e-3, batch, device, seed=300)
        mse_rows.append(
            {
                "experiment": "baselines",
                "model": baseline_name,
                "n": n,
                "test_mse": metrics["test_mse"],
                "ood_sparse_mse": _real_mse(baseline, splits["ood_sparse"], device),
                "ood_structured_mse": _real_mse(baseline, splits["ood_structured"], device),
                "params": count_parameters(baseline),
            }
        )

    _write_csv(out_dir / "mse_table.csv", mse_rows)
    _write_csv(out_dir / "diagonalization_error.csv", diag_rows)
    with (out_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"mode": mode, "device": str(device), "mse_rows": mse_rows, "diag_rows": diag_rows}, f, indent=2)
    return {"mse_rows": mse_rows, "diag_rows": diag_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fourier discovery experiments.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_all(args.mode, args.results_dir, args.device)


if __name__ == "__main__":
    main()
