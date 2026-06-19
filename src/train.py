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


def _apply_scale_augmentation(
    x: Tensor,
    h: Tensor,
    y: Tensor,
    scale_range: tuple[float, float] | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if scale_range is None:
        return x, h, y
    lo, hi = scale_range
    if lo <= 0.0 or hi < lo:
        raise ValueError("scale_range must be a positive (lo, hi) pair")
    shape = (x.shape[0], 1)
    log_lo = torch.log(torch.tensor(lo, device=x.device, dtype=x.dtype))
    log_hi = torch.log(torch.tensor(hi, device=x.device, dtype=x.dtype))
    sx = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(shape, device=x.device, dtype=x.dtype))
    sh = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(shape, device=x.device, dtype=x.dtype))
    return sx * x, sh * h, (sx * sh) * y


def _circulant_matrix_batch(h: Tensor) -> Tensor:
    if h.ndim != 2:
        raise ValueError(f"h must have shape (batch, n); got {tuple(h.shape)}")
    n = h.shape[1]
    rows = torch.arange(n, device=h.device).view(n, 1)
    cols = torch.arange(n, device=h.device).view(1, n)
    indices = (rows - cols) % n
    return h[:, indices]


def _optimizer_param_groups(
    model: nn.Module,
    lr: float,
    lr_basis: float | None,
    lr_interaction: float | None,
) -> list[dict[str, Any]]:
    if lr_basis is None and lr_interaction is None:
        return [{"params": list(model.parameters()), "lr": lr}]

    basis_params: list[nn.Parameter] = []
    interaction_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in {"A", "B"}:
            basis_params.append(param)
        elif name.startswith("g_phi."):
            interaction_params.append(param)
        else:
            other_params.append(param)

    groups: list[dict[str, Any]] = []
    if basis_params:
        groups.append({"params": basis_params, "lr": lr if lr_basis is None else lr_basis})
    if interaction_params:
        groups.append({"params": interaction_params, "lr": lr if lr_interaction is None else lr_interaction})
    if other_params:
        groups.append({"params": other_params, "lr": lr})
    return groups


def _normalize_complex_rows_(matrix: Tensor, target_norm: float, eps: float = 1e-12) -> None:
    norms = torch.linalg.norm(matrix, dim=1, keepdim=True).clamp_min(eps)
    matrix.mul_(target_norm / norms)


@torch.no_grad()
def _apply_gauge_row_norm_(model: nn.Module, gauge_row_norm: float | None) -> None:
    if gauge_row_norm is None:
        return
    for name in ("A", "B"):
        param = getattr(model, name, None)
        if isinstance(param, nn.Parameter):
            _normalize_complex_rows_(param, gauge_row_norm)


def _model_prediction_and_loss(
    model: nn.Module,
    x: Tensor,
    h: Tensor,
    y: Tensor,
    encoded_loss_weight: float,
) -> tuple[Tensor, Tensor, Tensor | None]:
    if encoded_loss_weight <= 0.0 or not all(hasattr(model, attr) for attr in ("A", "encode", "interact")):
        pred = model(x, h)
        return pred, complex_mse(pred, y), None

    u, v = model.encode(x, h)
    z = model.interact(u, v)
    pred = torch.linalg.solve(model.A, z.T).T
    output_loss = complex_mse(pred, y)
    y_complex = y.to(device=pred.device, dtype=torch.complex64)
    target_z = y_complex @ model.A.T
    encoded_loss = ((z - target_z).abs() ** 2).mean()
    return pred, output_loss + encoded_loss_weight * encoded_loss, encoded_loss


def _diagonalization_loss(model: nn.Module, h: Tensor, max_kernels: int | None) -> Tensor | None:
    A = getattr(model, "A", None)
    if A is None:
        return None
    kernels = h if max_kernels is None else h[:max_kernels]
    if kernels.numel() == 0:
        return None
    C = _circulant_matrix_batch(kernels.float()).to(dtype=torch.complex64)
    inv_a = torch.linalg.inv(A)
    M = torch.matmul(torch.matmul(A.unsqueeze(0), C), inv_a.unsqueeze(0))
    diag = torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))
    offdiag = M - diag
    total = (M.abs() ** 2).sum(dim=(-2, -1)).clamp_min(1e-12)
    return ((offdiag.abs() ** 2).sum(dim=(-2, -1)) / total).mean()


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
    lr_basis: float | None = None,
    lr_interaction: float | None = None,
    batch: int | None = None,
    log_every: int = 200,
    results_dir: str | Path = "results",
    run_name: str | None = None,
    device: str | torch.device | None = None,
    seed: int = 0,
    save: bool = True,
    grad_clip: float | None = None,
    restore_best: bool = False,
    encoded_loss_weight: float = 0.0,
    gauge_row_norm: float | None = None,
    scale_range: tuple[float, float] | None = None,
    mse_loss_weight: float = 1.0,
    diag_loss_weight: float = 0.0,
    diag_loss_kernels: int | None = 8,
) -> dict[str, Any]:
    """Train a complex spectral model with Adam and complex MSE."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch is not None and batch <= 0:
        raise ValueError("batch must be positive or None")

    torch.manual_seed(seed)
    train_device = get_device(str(device)) if device is not None else get_device()
    model.to(train_device)
    _apply_gauge_row_norm_(model, gauge_row_norm)
    train_moved = _move_dataset(train_ds, train_device)
    test_moved = _move_dataset(test_ds, train_device) if test_ds is not None else None

    optimizer = torch.optim.Adam(_optimizer_param_groups(model, lr, lr_basis, lr_interaction), lr=lr)
    num_train = train_moved["x"].shape[0]
    history: dict[str, Any] = {
        "step": [],
        "train_mse": [],
        "test_mse": [],
        "device": str(train_device),
        "lr": lr,
        "lr_basis": lr_basis,
        "lr_interaction": lr_interaction,
        "batch": batch,
        "grad_clip": grad_clip,
        "restore_best": restore_best,
        "encoded_loss_weight": encoded_loss_weight,
        "gauge_row_norm": gauge_row_norm,
        "scale_range": scale_range,
        "mse_loss_weight": mse_loss_weight,
        "diag_loss_weight": diag_loss_weight,
        "diag_loss_kernels": diag_loss_kernels,
        "encoded_mse": [],
        "diag_loss": [],
    }
    best_test = float("inf")
    best_state: dict[str, Tensor] | None = None

    started = time.time()
    for step in range(1, steps + 1):
        model.train()
        if batch is None:
            indices = None
        else:
            indices = torch.randint(num_train, (batch,), device=train_device)
        x, h, y = _batch(train_moved, indices)
        x, h, y = _apply_scale_augmentation(x, h, y, scale_range)

        optimizer.zero_grad(set_to_none=True)
        loss: Tensor | None = None
        if mse_loss_weight > 0.0:
            _, task_loss, _ = _model_prediction_and_loss(model, x, h, y, encoded_loss_weight)
            loss = mse_loss_weight * task_loss
        diag_loss = _diagonalization_loss(model, h, diag_loss_kernels) if diag_loss_weight > 0.0 else None
        if diag_loss is not None:
            weighted_diag = diag_loss_weight * diag_loss
            loss = weighted_diag if loss is None else loss + weighted_diag
        if loss is None:
            raise ValueError("at least one of mse_loss_weight or diag_loss_weight must be positive")
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        _apply_gauge_row_norm_(model, gauge_row_norm)

        if step == 1 or step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                train_pred, _, train_encoded = _model_prediction_and_loss(
                    model, train_moved["x"], train_moved["h"], train_moved["y"], encoded_loss_weight
                )
                train_loss = complex_mse(train_pred, train_moved["y"])
                if test_moved is None:
                    test_loss = torch.tensor(float("nan"), device=train_device)
                    logged_diag_loss = None
                else:
                    test_pred, _, _ = _model_prediction_and_loss(
                        model, test_moved["x"], test_moved["h"], test_moved["y"], encoded_loss_weight
                    )
                    test_loss = complex_mse(test_pred, test_moved["y"])
                    logged_diag_loss = (
                        _diagonalization_loss(model, test_moved["h"], diag_loss_kernels)
                        if diag_loss_weight > 0.0
                        else None
                    )
            history["step"].append(step)
            history["train_mse"].append(float(train_loss.detach().cpu()))
            history["test_mse"].append(float(test_loss.detach().cpu()))
            history["encoded_mse"].append(None if train_encoded is None else float(train_encoded.detach().cpu()))
            history["diag_loss"].append(None if logged_diag_loss is None else float(logged_diag_loss.detach().cpu()))
            test_value = float(test_loss.detach().cpu())
            if restore_best and test_value < best_test:
                best_test = test_value
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        model.to(train_device)
        history["best_test_mse"] = best_test

    history["elapsed_sec"] = time.time() - started
    history["final_train_mse"] = evaluate_loss(model, train_ds, train_device)
    history["final_test_mse"] = evaluate_loss(model, test_ds, train_device) if test_ds is not None else float("nan")

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
