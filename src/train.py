from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


Dataset = dict[str, Tensor]


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        requested = torch.device(prefer)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def complex_mse(pred: Tensor, target: Tensor) -> Tensor:
    target_complex = target.to(device=pred.device, dtype=torch.complex64)
    return ((pred - target_complex).abs() ** 2).mean()


def _move_dataset(ds: Dataset, device: torch.device) -> Dataset:
    return {key: value.to(device=device) for key, value in ds.items()}


def _batch(ds: Dataset, indices: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    if indices is None:
        return ds["x"], ds["h"], ds["y"]
    return ds["x"][indices], ds["h"][indices], ds["y"][indices]


@torch.no_grad()
def evaluate_loss(model: nn.Module, ds: Dataset, device: torch.device | str | None = None) -> float:
    if device is not None:
        eval_device = get_device(str(device))
    else:
        first_param = next(model.parameters(), None)
        if first_param is not None:
            eval_device = first_param.device
        else:
            first_buffer = next(model.buffers(), None)
            eval_device = first_buffer.device if first_buffer is not None else torch.device("cpu")
    model.eval()
    moved = _move_dataset(ds, eval_device)
    pred = model(moved["x"], moved["h"])
    return float(complex_mse(pred, moved["y"]).detach().cpu())


def train(
    model: nn.Module,
    train_ds: Dataset,
    test_ds: Dataset | None,
    steps: int = 4_000,
    lr: float = 1e-2,
    batch: int | None = None,
    log_every: int = 200,
    results_dir: str | Path = "results",
    run_name: str | None = None,
    device: str | torch.device | None = None,
    seed: int = 0,
    save: bool = True,
) -> dict[str, Any]:
    """Train a complex spectral model with Adam and complex MSE."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch is not None and batch <= 0:
        raise ValueError("batch must be positive or None")

    torch.manual_seed(seed)
    train_device = get_device(str(device)) if device is not None else get_device()
    model.to(train_device)
    train_moved = _move_dataset(train_ds, train_device)
    test_moved = _move_dataset(test_ds, train_device) if test_ds is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_train = train_moved["x"].shape[0]
    history: dict[str, Any] = {
        "step": [],
        "train_mse": [],
        "test_mse": [],
        "device": str(train_device),
        "lr": lr,
        "batch": batch,
    }

    started = time.time()
    for step in range(1, steps + 1):
        model.train()
        if batch is None:
            indices = None
        else:
            indices = torch.randint(num_train, (batch,), device=train_device)
        x, h, y = _batch(train_moved, indices)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x, h)
        loss = complex_mse(pred, y)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_moved["x"], train_moved["h"])
                train_loss = complex_mse(train_pred, train_moved["y"])
                if test_moved is None:
                    test_loss = torch.tensor(float("nan"), device=train_device)
                else:
                    test_pred = model(test_moved["x"], test_moved["h"])
                    test_loss = complex_mse(test_pred, test_moved["y"])
            history["step"].append(step)
            history["train_mse"].append(float(train_loss.detach().cpu()))
            history["test_mse"].append(float(test_loss.detach().cpu()))

    history["elapsed_sec"] = time.time() - started
    history["final_train_mse"] = history["train_mse"][-1]
    history["final_test_mse"] = history["test_mse"][-1]

    if save:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = run_name or f"{model.__class__.__name__}_{int(time.time())}"
        checkpoint = {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_class": model.__class__.__name__,
            "n": getattr(model, "n", None),
            "history": history,
        }
        torch.save(checkpoint, out_dir / f"{name}_model.pt")
        with (out_dir / f"{name}_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return history
