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
from analysis.metrics import dft_matrix, multiplication_probe
from src.data import make_dataset
from src.models import SpectralNeuralModel, SpectralNeuralModelUntied
from src.train import get_device, train


def _model_from_checkpoint_payload(payload: dict[str, Any], fallback_n: int) -> Any:
    model_class = payload.get("model_class", "SpectralNeuralModel")
    model_n = int(payload.get("n") or fallback_n)
    state = payload.get("model_state", {})
    g_weights = sorted(
        (key for key in state if key.startswith("g_phi.net.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[2]),
    )
    hidden = int(state[g_weights[0]].shape[0]) if g_weights else 64
    depth = max(1, len(g_weights) - 1) if g_weights else 3
    if model_class in {"SpectralNeuralModelUntied", "SpectralConvUntied"}:
        return SpectralNeuralModelUntied(model_n, hidden=hidden, depth=depth)
    return SpectralNeuralModel(model_n, hidden=hidden, depth=depth)


def _load_or_train(checkpoint: str | None, n: int, results_dir: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    if checkpoint is None:
        n_matches = sorted(results_dir.glob(f"spectral_neural_n{n}_*_model.pt"))
        any_matches = sorted(results_dir.glob("spectral_neural_n*_model.pt"))
        matches = n_matches or any_matches
        if matches:
            checkpoint = str(matches[0])

    if checkpoint is not None and Path(checkpoint).exists():
        payload = torch.load(checkpoint, map_location=device)
        model = _model_from_checkpoint_payload(payload, n)
        model.load_state_dict(payload["model_state"], strict=False)
        model.to(device)
        return model, payload

    train_ds = make_dataset(n, 1_024, "dense", seed=500)
    test_ds = make_dataset(n, 256, "dense", seed=501)
    model = SpectralNeuralModel(n, seed=502)
    history = train(
        model,
        train_ds,
        test_ds,
        steps=500,
        lr=1e-3,
        batch=256,
        log_every=100,
        results_dir=results_dir,
        run_name=f"visualize_spectral_neural_n{n}",
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


@torch.no_grad()
def _save_multiplication_probe(model: Any, out: Path) -> None:
    if not hasattr(model, "g_phi"):
        return
    device = next(model.parameters()).device
    n = getattr(model, "n")
    ds = make_dataset(n, 512, "dense", seed=710)
    probe = multiplication_probe(model, data=ds, num=4_096, seed=711, mode="encoded")

    u, v = model.encode(ds["x"].to(device), ds["h"].to(device))
    u = u.reshape(-1)[: probe["num"]]
    v = v.reshape(-1)[: probe["num"]]
    features = torch.stack([u.real, u.imag, v.real, v.imag], dim=-1)
    out_raw = model.g_phi(features)
    g_pred = torch.complex(out_raw[..., 0], out_raw[..., 1]).detach().cpu()
    scale = complex(probe["scale_real"], probe["scale_imag"])
    target = (torch.tensor(scale, dtype=torch.complex64) * (u.detach().cpu() * v.detach().cpu())).to(torch.complex64)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(target.real.numpy(), g_pred.real.numpy(), s=6, alpha=0.35)
    axes[0].set_xlabel("Re(c u v)")
    axes[0].set_ylabel("Re(g_phi)")
    axes[0].set_title("real part")
    axes[1].scatter(target.imag.numpy(), g_pred.imag.numpy(), s=6, alpha=0.35)
    axes[1].set_xlabel("Im(c u v)")
    axes[1].set_ylabel("Im(g_phi)")
    axes[1].set_title("imag part")
    fig.suptitle(f"multiplication probe residual={probe['rel_residual']:.3g}")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _save_noise_curve(diag_csv: Path, probe_csv: Path, out: Path) -> None:
    diag_by_snr: dict[float, float] = {}
    if diag_csv.exists():
        with diag_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("experiment") == "noise_sweep" and row.get("snr_db") and row.get("diag_residual"):
                    diag_by_snr[float(row["snr_db"])] = float(row["diag_residual"])

    probe_by_snr: dict[float, float] = {}
    if probe_csv.exists():
        with probe_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("experiment") == "noise_sweep" and row.get("snr_db") and row.get("rel_residual"):
                    probe_by_snr[float(row["snr_db"])] = float(row["rel_residual"])

    snrs = sorted(set(diag_by_snr) | set(probe_by_snr), reverse=True)
    if not snrs:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    if diag_by_snr:
        ax.plot([s for s in snrs if s in diag_by_snr], [diag_by_snr[s] for s in snrs if s in diag_by_snr], marker="o", label="diagonalization")
    if probe_by_snr:
        ax.plot([s for s in snrs if s in probe_by_snr], [probe_by_snr[s] for s in snrs if s in probe_by_snr], marker="s", label="multiplication probe")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("residual")
    ax.set_title("Noise degradation")
    ax.invert_xaxis()
    ax.legend()
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
    _save_multiplication_probe(model, results_dir / "multiplication_probe.png")
    _save_noise_curve(
        results_dir / "diagonalization_error.csv",
        results_dir / "multiplication_probe.csv",
        results_dir / "noise_degradation.png",
    )


if __name__ == "__main__":
    main()
