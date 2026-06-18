from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from analysis.metrics import (
    dft_alignment_error,
    mean_diagonalization_residual,
    multiplication_probe,
    unitarity_stats,
)
from src.baselines import (
    BilinearTensor,
    HardProductLearnedBasis,
    HardProductOracle,
    MLPControl,
    NonlinearEncoderControl,
    count_parameters,
)
from src.data import Dataset, standard_splits
from src.models import SpectralNeuralModel, SpectralNeuralModelUntied
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


def _probe_row(experiment: str, model_name: str, n: int, model: nn.Module, ds: Dataset, seed: int, **extra: Any) -> dict[str, Any]:
    probe = multiplication_probe(model, data=ds, num=4_096, seed=seed, mode="encoded")
    return {
        "experiment": experiment,
        "model": model_name,
        "n": n,
        "probe_mode": probe["mode"],
        "probe_num": probe["num"],
        "rel_residual": probe["rel_residual"],
        "scale_real": probe["scale_real"],
        "scale_imag": probe["scale_imag"],
        "scale_abs": probe["scale_abs"],
        **extra,
    }


def _row_scaled_relative(A: Tensor, B: Tensor) -> float:
    A_cpu = A.detach().cpu().to(torch.complex64)
    B_cpu = B.detach().cpu().to(torch.complex64)
    residuals: list[Tensor] = []
    for row in range(A_cpu.shape[0]):
        denom = torch.vdot(B_cpu[row], B_cpu[row])
        if denom.abs() < 1e-12:
            residuals.append(torch.tensor(1.0))
            continue
        scale = torch.vdot(B_cpu[row], A_cpu[row]) / denom
        residuals.append(torch.linalg.norm(scale * B_cpu[row] - A_cpu[row]) / torch.linalg.norm(A_cpu[row]).clamp_min(1e-12))
    return float(torch.stack(residuals).mean())


def _variance_y(ds: Dataset) -> float:
    return float(ds["y"].var(unbiased=False))


def _r2_from_mse(mse: float, ds: Dataset) -> float:
    var_y = _variance_y(ds)
    if var_y <= 1e-12:
        return 0.0
    return 1.0 - mse / var_y


def _train_and_score(
    model: nn.Module,
    train_ds: Dataset,
    test_ds: Dataset,
    steps: int,
    lr: float,
    batch: int | None,
    results_dir: Path,
    run_name: str,
    device: torch.device,
    seed: int,
    lr_basis: float | None = None,
    lr_interaction: float | None = None,
    grad_clip: float | None = 10.0,
    restore_best: bool = True,
    encoded_loss_weight: float = 0.0,
    gauge_row_norm: float | None = None,
    scale_range: tuple[float, float] | None = None,
    diag_loss_weight: float = 0.0,
    diag_loss_kernels: int | None = 8,
) -> dict[str, Any]:
    history = train(
        model,
        train_ds,
        test_ds,
        steps=steps,
        lr=lr,
        lr_basis=lr_basis,
        lr_interaction=lr_interaction,
        batch=batch,
        log_every=max(1, steps // 10),
        results_dir=results_dir,
        run_name=run_name,
        device=device,
        seed=seed,
        grad_clip=grad_clip,
        restore_best=restore_best,
        encoded_loss_weight=encoded_loss_weight,
        gauge_row_norm=gauge_row_norm,
        scale_range=scale_range,
        diag_loss_weight=diag_loss_weight,
        diag_loss_kernels=diag_loss_kernels,
    )
    return {
        "history": history,
        "test_mse": evaluate_loss(model, test_ds, device),
    }


def _save_checkpoint(model: nn.Module, history: dict[str, Any], results_dir: Path, run_name: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_class": model.__class__.__name__,
        "n": getattr(model, "n", None),
        "history": history,
    }
    torch.save(checkpoint, results_dir / f"{run_name}_model.pt")
    with (results_dir / f"{run_name}_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _discovery_seed(n: int) -> int:
    return 4 if n <= 16 else 6


def _discovery_phase_config(n: int, mode: str) -> dict[str, Any]:
    if mode == "smoke":
        return {
            "phase1_steps": 1_000,
            "phase2_steps": 1_000,
            "phase3_steps": 500,
            "diag_loss_weight": 20.0 if n < 64 else 100.0,
            "diag_loss_kernels": min(16, n),
            "gauge_row_norm": 1.0,
        }
    if n >= 64:
        return {
            "phase1_steps": 30_000,
            "phase2_steps": 80_000,
            "phase3_steps": 30_000,
            "diag_loss_weight": 100.0,
            "diag_loss_kernels": 32,
            "gauge_row_norm": 1.0,
        }
    return {
        "phase1_steps": 15_000,
        "phase2_steps": 40_000,
        "phase3_steps": 10_000,
        "diag_loss_weight": 20.0,
        "diag_loss_kernels": 16,
        "gauge_row_norm": 1.0,
    }


def _train_discovery_and_score(
    model: SpectralNeuralModel,
    train_ds: Dataset,
    test_ds: Dataset,
    batch: int | None,
    results_dir: Path,
    run_name: str,
    device: torch.device,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    cfg = _discovery_phase_config(model.n, mode)
    phase1 = train(
        model,
        train_ds,
        test_ds,
        steps=cfg["phase1_steps"],
        lr=1e-3,
        batch=batch,
        log_every=max(1, cfg["phase1_steps"] // 3),
        device=device,
        seed=seed,
        save=False,
        restore_best=True,
        diag_loss_weight=cfg["diag_loss_weight"],
        diag_loss_kernels=cfg["diag_loss_kernels"],
        gauge_row_norm=cfg["gauge_row_norm"],
    )

    model.A.requires_grad_(False)
    phase2 = train(
        model,
        train_ds,
        test_ds,
        steps=cfg["phase2_steps"],
        lr=1e-3,
        batch=batch,
        log_every=max(1, cfg["phase2_steps"] // 4),
        device=device,
        seed=seed + 1_000,
        save=False,
        restore_best=True,
        gauge_row_norm=cfg["gauge_row_norm"],
    )
    phase3 = train(
        model,
        train_ds,
        test_ds,
        steps=cfg["phase3_steps"],
        lr=3e-4,
        batch=batch,
        log_every=max(1, cfg["phase3_steps"] // 2),
        device=device,
        seed=seed + 2_000,
        save=False,
        restore_best=True,
        gauge_row_norm=cfg["gauge_row_norm"],
    )
    model.A.requires_grad_(True)

    final_train = evaluate_loss(model, train_ds, device)
    final_test = evaluate_loss(model, test_ds, device)
    history: dict[str, Any] = {
        "strategy": "diag_regularized_basis_then_frozen_interaction",
        "device": str(device),
        "batch": batch,
        "config": cfg,
        "phases": {"basis": phase1, "interaction": phase2, "interaction_finetune": phase3},
        "elapsed_sec": phase1["elapsed_sec"] + phase2["elapsed_sec"] + phase3["elapsed_sec"],
        "final_train_mse": final_train,
        "final_test_mse": final_test,
        "step": phase1["step"] + [cfg["phase1_steps"] + s for s in phase2["step"]] + [
            cfg["phase1_steps"] + cfg["phase2_steps"] + s for s in phase3["step"]
        ],
        "train_mse": phase1["train_mse"] + phase2["train_mse"] + phase3["train_mse"],
        "test_mse": phase1["test_mse"] + phase2["test_mse"] + phase3["test_mse"],
    }
    _save_checkpoint(model, history, results_dir, run_name)
    return {"history": history, "test_mse": final_test}


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
        baseline_steps = 700
        batch = 128
        noise_snrs = [30.0]
    elif mode == "full":
        n_values = [16, 32, 64]
        train_num, test_num, ood_num = 20_000, 4_000, 4_000
        baseline_steps = 4_000
        batch = 512
        noise_snrs = [40.0, 30.0, 20.0, 10.0, 0.0]
    else:
        raise ValueError("mode must be 'smoke' or 'full'")

    mse_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []

    for n in n_values:
        splits = standard_splits(n, train_num, test_num, ood_num, seed=100 + n)
        model = SpectralNeuralModel(n, hidden=128, seed=_discovery_seed(n))
        run_name = f"spectral_neural_n{n}_{mode}"
        scored = _train_discovery_and_score(
            model,
            splits["train"],
            splits["test"],
            batch,
            out_dir,
            run_name,
            device,
            seed=10 + n,
            mode=mode,
        )
        sparse_mse = evaluate_loss(model, splits["ood_sparse"], device)
        structured_mse = evaluate_loss(model, splits["ood_structured"], device)
        mse_rows.append(
            {
                "experiment": "size_sweep",
                "model": "SpectralNeuralModel",
                "n": n,
                "test_mse": scored["test_mse"],
                "ood_sparse_mse": sparse_mse,
                "ood_structured_mse": structured_mse,
                "train_seconds": scored["history"]["elapsed_sec"],
            }
        )
        diag_rows.append(
            {
                "experiment": "size_sweep",
                "model": "SpectralNeuralModel",
                "n": n,
                **_spectral_summary(model, splits["test"]),
            }
        )
        probe_rows.append(_probe_row("size_sweep", "SpectralNeuralModel", n, model, splits["test"], seed=200 + n))

    n = 32 if mode == "full" else 16
    splits = standard_splits(n, train_num, test_num, ood_num, seed=232)

    for snr in noise_snrs:
        noisy_train = _noisy_dataset(splits["train"], snr, seed=int(1_000 + snr))
        model = SpectralNeuralModel(n, hidden=128, seed=_discovery_seed(n))
        _train_discovery_and_score(
            model,
            noisy_train,
            splits["test"],
            batch,
            out_dir,
            f"noise_snr{int(snr)}_spectral_neural_n{n}_{mode}",
            device,
            seed=30 + int(snr),
            mode=mode,
        )
        mse_rows.append(
            {
                "experiment": "noise_sweep",
                "model": "SpectralNeuralModel",
                "n": n,
                "snr_db": snr,
                "test_mse": evaluate_loss(model, splits["test"], device),
            }
        )
        diag_rows.append(
            {
                "experiment": "noise_sweep",
                "model": "SpectralNeuralModel",
                "n": n,
                "snr_db": snr,
                **_spectral_summary(model, splits["test"]),
            }
        )
        probe_rows.append(_probe_row("noise_sweep", "SpectralNeuralModel", n, model, splits["test"], seed=300 + int(snr), snr_db=snr))

    untied = SpectralNeuralModelUntied(n, hidden=128, seed=_discovery_seed(n))
    _train_discovery_and_score(
        untied,
        splits["train"],
        splits["test"],
        batch,
        out_dir,
        f"untied_spectral_neural_n{n}_{mode}",
        device,
        seed=40,
        mode=mode,
    )
    mse_rows.append(
        {
            "experiment": "untied_variant",
            "model": "SpectralNeuralModelUntied",
            "n": n,
            "test_mse": evaluate_loss(untied, splits["test"], device),
            "row_scaled_A_B_relative": _row_scaled_relative(untied.A, untied.B),
        }
    )
    diag_rows.append(
        {
            "experiment": "untied_variant",
            "model": "SpectralNeuralModelUntied",
            "n": n,
            **_spectral_summary(untied, splits["test"]),
        }
    )
    probe_rows.append(_probe_row("untied_variant", "SpectralNeuralModelUntied", n, untied, splits["test"], seed=404))

    oracle = HardProductOracle(n).to(device)
    oracle_test = evaluate_loss(oracle, splits["test"], device)
    mse_rows.append(
        {
            "experiment": "baselines",
            "model": "HardProductOracle",
            "n": n,
            "test_mse": oracle_test,
            "ood_sparse_mse": evaluate_loss(oracle, splits["ood_sparse"], device),
            "ood_structured_mse": evaluate_loss(oracle, splits["ood_structured"], device),
            "params": count_parameters(oracle),
        }
    )

    baseline_specs: list[tuple[str, nn.Module, float]] = [
        ("HardProductLearnedBasis", HardProductLearnedBasis(n, seed=5), 1e-2),
        ("BilinearTensor", BilinearTensor(n, seed=51), 1e-2),
        ("MLPControl", MLPControl(n), 1e-3),
        ("NonlinearEncoderControl", NonlinearEncoderControl(n, seed=52), 1e-3),
    ]
    for idx, (name, baseline, lr) in enumerate(baseline_specs):
        scored = _train_and_score(
            baseline,
            splits["train"],
            splits["test"],
            baseline_steps,
            lr,
            batch,
            out_dir,
            f"baseline_{name}_n{n}_{mode}",
            device,
            seed=500 + idx,
        )
        test_mse = scored["test_mse"]
        mse_rows.append(
            {
                "experiment": "baselines",
                "model": name,
                "n": n,
                "test_mse": test_mse,
                "ood_sparse_mse": evaluate_loss(baseline, splits["ood_sparse"], device),
                "ood_structured_mse": evaluate_loss(baseline, splits["ood_structured"], device),
                "r2": _r2_from_mse(test_mse, splits["test"]),
                "var_y": _variance_y(splits["test"]),
                "params": count_parameters(baseline),
            }
        )
        if hasattr(baseline, "A"):
            diag_rows.append(
                {
                    "experiment": "baselines",
                    "model": name,
                    "n": n,
                    **_spectral_summary(baseline, splits["test"]),
                }
            )
        if hasattr(baseline, "g_phi"):
            probe_rows.append(_probe_row("baselines", name, n, baseline, splits["test"], seed=600 + idx))

    _write_csv(out_dir / "mse_table.csv", mse_rows)
    _write_csv(out_dir / "diagonalization_error.csv", diag_rows)
    _write_csv(out_dir / "multiplication_probe.csv", probe_rows)
    summary = {
        "mode": mode,
        "device": str(device),
        "mse_rows": mse_rows,
        "diag_rows": diag_rows,
        "probe_rows": probe_rows,
    }
    with (out_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return {"mse_rows": mse_rows, "diag_rows": diag_rows, "probe_rows": probe_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fourier discovery experiments.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_all(args.mode, args.results_dir, args.device)


if __name__ == "__main__":
    main()
