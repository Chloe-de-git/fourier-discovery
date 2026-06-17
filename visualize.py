from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

_DEFAULT_MPLCONFIGDIR = Path(__file__).resolve().parent / "results" / ".mplconfig"
_DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from analysis.alignment import row_assignment, scaled_row_residual
from analysis.metrics import dft_matrix
from src.data import make_dataset
from src.models import SpectralConvTied, SpectralConvUntied
from src.train import get_device, train


def _load_or_train(checkpoint: str | None, n: int, results_dir: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    if checkpoint is not None and Path(checkpoint).exists():
        payload = torch.load(checkpoint, map_location=device)
        model_class = payload.get("model_class", "SpectralConvUntied")
        model_n = int(payload.get("n") or n)
        model = SpectralConvTied(model_n) if model_class == "SpectralConvTied" else SpectralConvUntied(model_n)
        model.load_state_dict(payload["model_state"])
        model.to(device)
        return model, payload

    train_ds = make_dataset(n, 1_024, "dense", seed=500)
    test_ds = make_dataset(n, 256, "dense", seed=501)
    model = SpectralConvUntied(n, seed=502)
    history = train(
        model,
        train_ds,
        test_ds,
        steps=500,
        lr=1e-2,
        batch=256,
        log_every=100,
        results_dir=results_dir,
        run_name=f"visualize_auto_n{n}",
        device=device,
        seed=503,
    )
    return model, {"history": history, "n": n}


def _save_abs_heatmap(A: torch.Tensor, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A.detach().cpu().abs().numpy(), aspect="auto", cmap="viridis")
    ax.set_title("|A| heatmap")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _save_alignment(A: torch.Tensor, out: Path, rows_to_show: int = 4) -> None:
    A_cpu = A.detach().cpu().to(torch.complex64)
    F = dft_matrix(A_cpu.shape[0])
    row_ind, col_ind, _ = row_assignment(A_cpu, F)
    count = min(rows_to_show, A_cpu.shape[0])
    fig, axes = plt.subplots(count, 2, figsize=(9, 2.4 * count), squeeze=False)
    for idx in range(count):
        row = int(row_ind[idx])
        col = int(col_ind[idx])
        _, scale = scaled_row_residual(A_cpu[row], F[col])
        aligned = scale * A_cpu[row]
        axes[idx, 0].plot(aligned.abs().numpy(), label="learned")
        axes[idx, 0].plot(F[col].abs().numpy(), linestyle="--", label="DFT")
        axes[idx, 0].set_ylabel(f"row {row}->{col}")
        axes[idx, 0].set_title("magnitude")
        axes[idx, 1].plot(torch.angle(aligned).numpy(), label="learned")
        axes[idx, 1].plot(torch.angle(F[col]).numpy(), linestyle="--", label="DFT")
        axes[idx, 1].set_title("phase")
    axes[0, 0].legend()
    axes[0, 1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _save_spectrum(model: Any, out: Path) -> None:
    A = model.A.detach().cpu().to(torch.complex64)
    n = A.shape[0]
    h = make_dataset(n, 1, "dense", seed=700)["h"][0]
    F = dft_matrix(n)
    true = F @ h.to(torch.complex64)
    if hasattr(model, "B"):
        learned = model.B.detach().cpu().to(torch.complex64) @ h.to(torch.complex64)
    else:
        learned = model.A.detach().cpu().to(torch.complex64) @ h.to(torch.complex64)

    row_ind, col_ind, _ = row_assignment(A, F)
    perm = torch.empty(n, dtype=torch.long)
    for row, col in zip(row_ind, col_ind, strict=True):
        perm[int(row)] = int(col)
    true_ordered = true[perm]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(learned.abs().numpy(), label="learned")
    axes[0].plot(true_ordered.abs().numpy(), linestyle="--", label="DFT")
    axes[0].set_title("multiplier magnitude")
    axes[1].plot(torch.angle(learned).numpy(), label="learned")
    axes[1].plot(torch.angle(true_ordered).numpy(), linestyle="--", label="DFT")
    axes[1].set_title("multiplier phase")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _save_noise_curve(csv_path: Path, out: Path) -> None:
    if not csv_path.exists():
        return
    snr: list[float] = []
    residual: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("experiment") == "noise_sweep" and row.get("snr_db") and row.get("diag_residual"):
                snr.append(float(row["snr_db"]))
                residual.append(float(row["diag_residual"]))
    if not snr:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snr, residual, marker="o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("diagonalization residual")
    ax.set_title("Noise degradation")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Fourier discovery figures.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    model, _ = _load_or_train(args.checkpoint, args.n, results_dir, device)
    A = model.A.detach().cpu()

    _save_abs_heatmap(A, results_dir / "A_abs_heatmap.png")
    _save_alignment(A, results_dir / "alignment_rows.png")
    _save_spectrum(model, results_dir / "spectrum_match.png")
    _save_noise_curve(results_dir / "diagonalization_error.csv", results_dir / "noise_degradation.png")


if __name__ == "__main__":
    main()
