# SPEC: Learning a Linear Transform that Diagonalizes Circular Convolution

## Task

Build a PyTorch project that **learns**, from random data only, a linear transform `A` such that circular convolution becomes coordinate-wise multiplication. The model fits triples `(x, h, y)` where `y` is the circular convolution of `x` and `h`, using:

```
ŷ = A⁻¹ ( (A x) ⊙ (B h) )
```

`A`, `B` are free learnable complex matrices (no imposed structure). After training, analyze whether `A` diagonalizes all convolution operators and matches the DFT matrix up to gauge.

---

## Hard constraints (must follow exactly)

1. In `src/data.py`, `src/convolution.py`, `src/models.py`, `src/train.py`:
   **do NOT use any FFT or built-in convolution.** Forbidden: `torch.fft.*`, `numpy.fft.*`, `scipy.fft`, `scipy.fftpack`, `torch.nn.functional.conv1d/conv2d`, `numpy.convolve`, `scipy.signal.*`. Circular convolution must be implemented from its definition (explicit sum or `roll`).
2. The learnable matrices `A`, `B` must be **unconstrained complex matrices** (`torch.complex64`). Do not parameterize them as unitary, circulant, real, symmetric, or anything structured.
3. The true DFT matrix may be constructed **only inside `analysis/` and `src/baselines.py`** (for comparison / oracle). It must never be imported or referenced in `data.py`, `convolution.py`, `models.py`, or `train.py`.
4. Use `torch.complex64` throughout the model and loss.

---

## Tech stack

Python 3.10+, `torch`, `numpy`, `scipy` (for `scipy.optimize.linear_sum_assignment`), `matplotlib`. Add a `requirements.txt`.

---

## Definitions

All signals are length `n`. Batches use shape `(batch, n)`.

**Circular convolution** (`x`, `h` real, batched `(batch, n)`):
```
y[b, t] = Σ_{s=0..n-1} x[b, s] · h[b, (t - s) mod n]
```
Output `y` is real, shape `(batch, n)`.

**Circulant matrix** of a single kernel `h` (length `n`):
```
C[t, s] = h[(t - s) mod n]        # shape (n, n)
```
Must satisfy `C @ x == circular_conv(x, h)` for a single `x`.

**Model (untied)** — parameters `A, B ∈ ℂ^{n×n}`. Given real batched `x, h`:
```
X = x.to(complex64);  H = h.to(complex64)        # (batch, n)
Xf = X @ A.T                                      # (A x) per row
Hf = H @ B.T                                      # (B h) per row
P  = Xf * Hf                                       # elementwise, (batch, n)
ŷ  = P @ torch.linalg.inv(A).T                    # apply A⁻¹ per row
```

**Model (tied)**: same as above with `B = A` (only `A` is a parameter).

**Loss**: complex MSE against `y` embedded as complex with zero imaginary part:
```
loss = mean( (ŷ - (y + 0j)).abs() ** 2 )
```
(This penalizes both real-part error and spurious imaginary part.)

---

## Repository structure

```
fourier-discovery/
├── requirements.txt
├── README.md
├── src/
│   ├── data.py
│   ├── convolution.py
│   ├── models.py
│   ├── train.py
│   ├── baselines.py
│   └── experiments.py
├── analysis/
│   ├── metrics.py
│   └── alignment.py
├── visualize.py
├── tests/
│   └── test_sanity.py
└── results/            # generated: csv + png
```

---

## Module specs

### `src/convolution.py`
- `circular_conv(x: Tensor, h: Tensor) -> Tensor`
  - `x, h`: real, shape `(batch, n)` (also accept `(n,)` and treat as batch 1).
  - Returns real `(batch, n)`. Implement by definition (e.g. sum of `torch.roll`-shifted products). No FFT.
- `circulant_matrix(h: Tensor) -> Tensor`
  - `h`: real, shape `(n,)`. Returns real `(n, n)` with `C[t, s] = h[(t - s) mod n]`.

### `src/data.py`
- `make_dataset(n: int, num: int, kernel_dist: str = "dense", seed: int = 0) -> dict`
  - Returns `{"x": (num, n) float32, "h": (num, n) float32, "y": (num, n) float32}`.
  - `x`: standard normal.
  - `h` depends on `kernel_dist`:
    - `"dense"`: standard normal.
    - `"sparse"`: exactly `k` nonzero entries (e.g. `k = max(2, n//8)`) at random positions, normal values, rest zero.
    - `"structured"`: random but with smoothly decaying magnitude across the index (e.g. multiply normal noise by a random decaying envelope). Do not describe this in frequency terms; just produce a different-looking distribution.
  - `y` computed via `circular_conv`. No FFT.
- Convenience: a function returning a standard split — `train` (dense), `test` (dense, different seed), `ood` (sparse and/or structured).

### `src/models.py`
- `class SpectralConvUntied(nn.Module)`: params `A, B` complex `(n, n)`; init as small random complex (or `identity + small complex noise`) so `A` is invertible; `forward(x, h) -> ŷ` per the untied formula.
- `class SpectralConvTied(nn.Module)`: single param `A`; `B = A`.
- Both expose the learned matrices via attributes `A` (and `B`).
- **No DFT, no FFT, no structural constraint here.**

### `src/train.py`
- `train(model, train_ds, test_ds, steps=4000, lr=1e-2, batch=None, log_every=200) -> history`
  - Adam optimizer. Complex MSE loss. Full-batch if `batch is None`, else minibatch.
  - Cast inputs to complex inside the model, not in the dataset.
  - Log train/test MSE; return `history` dict with loss curves and final metrics.
  - Save model state and history to `results/`.

### `src/baselines.py`
- `class FixedDFTOracle(nn.Module)`: builds the true DFT matrix `F[j,k] = exp(-2πi·j·k/n)` internally, sets `A = B = F`, runs the same forward formula. No learnable params. Used to show the architecture can reproduce `y` exactly (sanity) and to provide a loss reference.
- `class MLPBaseline(nn.Module)`: `ŷ = MLP([x ; h])`, real-valued, a few hidden layers. For comparison only.
- `class CNNBaseline(nn.Module)`: a small 1-D conv net mapping `[x ; h] -> ŷ`. For comparison only. (This baseline may use `conv1d`; the FFT ban is about the ground-truth/main model, not the explicitly-labeled CNN baseline. Document this clearly.)

### `analysis/metrics.py`
Construct the true DFT matrix here only.
- `dft_matrix(n) -> Tensor` complex `(n, n)`, `F[j,k] = exp(-2πi·j·k/n)`.
- `diagonalization_residual(A: Tensor, h: Tensor) -> float`
  - `C = circulant_matrix(h).to(complex)`, `M = A @ C @ torch.linalg.inv(A)`.
  - return `offdiag_energy / total_energy = Σ_{i≠j}|M_ij|² / Σ_{ij}|M_ij|²`.
- `mean_diagonalization_residual(A, kernels: Tensor) -> float`: average over many random `h`.
- `dft_alignment_error(A: Tensor) -> dict`
  - `F = dft_matrix(n)`. Row-normalize both `A` and `F` to unit L2 norm.
  - Cost matrix `cost[i,j] = -|⟨A_hat[i], F_hat[j]⟩|`; solve assignment with `linear_sum_assignment`.
  - For each matched pair `(i, j)`, optimal complex scale `c = ⟨F[j], A[i]⟩ / ⟨A[i], A[i]⟩`; per-row relative residual `‖c·A[i] − F[j]‖ / ‖F[j]‖`.
  - Return `{"perm": ..., "mean_row_residual": ..., "mean_abs_corr": ...}`.
- `spectrum_match(model, h: Tensor) -> dict`
  - Learned multiplier: untied `B @ h`; or `diag(A @ circulant_matrix(h).to(complex) @ inv(A))`.
  - True spectrum `F @ h`. Compare as multisets: apply the permutation from `dft_alignment_error`, or compare magnitude-sorted; return relative error.
- `unitarity_stats(A: Tensor) -> dict`
  - `|A_ij|` mean and (std / mean); and `‖AᴴA − cI‖_F / ‖cI‖_F` with `c = trace(AᴴA)/n`.

### `analysis/alignment.py`
Helper(s) for the assignment + per-row complex scaling used by `dft_alignment_error` (factor out if reused).

### `src/experiments.py`
Runnable script(s) that produce the `results/` artifacts:
- For `n in [16, 32, 64]`: train untied; record test MSE, mean diagonalization residual, DFT alignment error, unitarity stats.
- Noise sweep: add Gaussian noise to `y` at several SNRs; train/evaluate; record how diagonalization residual degrades.
- Generalization: train on `dense`, evaluate on `ood` (sparse/structured); record OOD MSE and diagonalization residual.
- tied vs untied at `n=32`: both should converge; for untied also report `‖A − B‖ / ‖A‖` after gauge alignment.
- Baselines at `n=32`: FixedDFTOracle loss, MLP/CNN test MSE + OOD MSE + param counts.
- Write `results/mse_table.csv`, `results/diagonalization_error.csv`.

### `visualize.py`
- `|A|` heatmap (expected: near-uniform magnitude).
- Alignment figure: matched rows of `A` vs `F` (magnitude and phase) after gauge alignment.
- Spectrum figure: learned vs true multiplier for one `h`.
- Noise-degradation curve: diagonalization residual vs SNR.
- Save PNGs to `results/`.

### `tests/test_sanity.py`
- `circulant_matrix(h) @ x` matches `circular_conv(x, h)` (random `h`, `x`).
- `circular_conv` is commutative: `conv(x,h) ≈ conv(h,x)`.
- `FixedDFTOracle` reproduces `y` to MSE `< 1e-8`.
- After a short untied training run at `n=16`, test MSE `< 1e-3` and mean diagonalization residual `< 1e-2`.

---

## Recommended defaults

- `n = 32` primary; also `16`, `64`.
- Train set `num = 10000` (dense), test `num = 2000`, ood `num = 2000`.
- `steps = 4000`, `lr = 1e-2`, Adam, full-batch (small `n`); raise steps if loss plateaus high.
- Init `A` (and `B`) `= eye(n) + 0.01 * (randn + i·randn)` cast to complex64.

---

## Acceptance criteria

1. `tests/test_sanity.py` passes.
2. Untied model at `n=32`: test MSE on the same order as `FixedDFTOracle` loss (both near machine-noise small).
3. `mean_diagonalization_residual(A, random kernels)` ≈ 0 (e.g. `< 1e-2`).
4. After gauge alignment, `dft_alignment_error` `mean_row_residual` is small (e.g. `< 0.1`); `|A_ij|` is near-uniform (`std/mean` small).
5. OOD evaluation (train dense → test sparse/structured) keeps low MSE and low diagonalization residual.
6. `results/` contains the CSV tables and PNG figures listed above.