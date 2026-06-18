# SPEC: Discovering the Fourier Basis and the Multiplication Law via a Neural Interaction Module

## Task

Build a PyTorch project that, from random circular-convolution data only, **discovers**
(a) a linear basis in which convolution decouples, which turns out to be the DFT, and
(b) that the per-coordinate interaction is complex multiplication.

The model fits triples `(x, h, y)` with `y = circular_conv(x, h)`, using:

```
u = A x ,  v = A h                 # A: learnable complex linear transform (n×n), shared (tied)
z[k] = g_phi( u[k], v[k] )         # g_phi: a small MLP, applied coordinate-wise, shared weights
ŷ = A⁻¹ z
```

The **only structural assumption** is: there exists *some* linear basis in which the interaction
is **coordinate-independent** (each output coordinate `z[k]` depends only on the same-index pair
`(u[k], v[k])`). We do **not** assume the basis is Fourier, and we do **not** assume the
interaction is multiplication. Both must emerge from training:
- `A` is forced to the DFT (up to gauge): coordinate-independence requires `A` to diagonalize all
  circulant operators, and the only basis that simultaneously diagonalizes them is the DFT up to
  permutation/phase/scaling.
- `g_phi` is forced to complex multiplication (up to one global complex scale tied to A's scale).

---

## Hard constraints (must follow exactly)

1. In `src/data.py`, `src/convolution.py`, `src/models.py`, `src/train.py`:
   **NO FFT or built-in convolution.** Forbidden: `torch.fft.*`, `numpy.fft.*`, `scipy.fft`,
   `scipy.fftpack`, `torch.nn.functional.conv1d/conv2d`, `numpy.convolve`, `scipy.signal.*`.
   Circular convolution is implemented from its definition (explicit sum or `roll`).
2. The encoder `A` is an **unconstrained complex matrix** (`torch.complex64`). Do NOT parameterize
   it as unitary / circulant / real / symmetric / DFT-like.
3. `g_phi` is a **generic MLP** (with nonlinearity). Do NOT hard-code multiplication, do NOT make it
   bilinear-by-construction. It must be free to learn any function of `(u[k], v[k])`.
4. The crucial structural choice — and the ONLY one allowed — is that `g_phi` sees **only the
   same-index pair** `(u[k], v[k])`, with **weights shared across all coordinates k**. It must NOT
   see other coordinates.
5. The true DFT matrix may be constructed **only inside `analysis/` and `src/baselines.py`**
   (for comparison / oracle). Never reference it in `data.py`, `convolution.py`, `models.py`, `train.py`.
6. Use `torch.complex64` for `A`, `u`, `v`, `z`, `ŷ`, and the loss.

---

## Tech stack

Python 3.10+, `torch`, `numpy`, `scipy` (`scipy.optimize.linear_sum_assignment`), `matplotlib`.
Add `requirements.txt`.

---

## Definitions

Signals are length `n`. Batches use shape `(batch, n)`.

**Circular convolution** (`x`, `h` real, batched `(batch, n)`):
```
y[b, t] = Σ_{s=0..n-1} x[b, s] · h[b, (t - s) mod n]
```
Output real `(batch, n)`. Implement via `roll`/explicit sum. No FFT.

**Circulant matrix** of a single kernel `h` (length `n`):
```
C[t, s] = h[(t - s) mod n]        # (n, n), satisfies C @ x == circular_conv(x, h)
```

**Model forward (tied A)** — parameter `A ∈ ℂ^{n×n}`, plus the MLP `g_phi`:
```
X = x.to(complex64);  H = h.to(complex64)          # (batch, n)
u = X @ A.T                                          # (batch, n) complex
v = H @ A.T                                          # (batch, n) complex
# coordinate-wise interaction:
feat = stack([u.real, u.imag, v.real, v.imag], dim=-1)   # (batch, n, 4) real
out  = g_phi(feat)                                       # (batch, n, 2) real  -> [Re z, Im z]
z    = complex(out[..., 0], out[..., 1])                 # (batch, n) complex
ŷ    = z @ torch.linalg.inv(A).T                         # (batch, n) complex
```
`g_phi` is an MLP `ℝ^4 → ℝ^2` (e.g. 2–3 hidden layers, width 64, tanh or ReLU), applied to the last
dim, i.e. independently at every `(batch, n)` location with shared weights.

**Loss** (complex MSE against `y` embedded as complex):
```
loss = mean( (ŷ - (y + 0j)).abs() ** 2 )
```

---

## Why the structure forces the discovery (for context; do not encode as constraints)

If `ŷ = A⁻¹ g(Ax, Ah)` reproduces `x*h` for all `x,h` with `g` coordinate-wise, then `A` must turn
every circulant `C_h` into a coordinate-wise (diagonal) action. The circulant algebra is generated
by the cyclic shift, whose eigenvalues are distinct, so the simultaneous eigenbasis is unique up to
permutation and per-coordinate scaling = the DFT up to gauge. Given that basis, the only
coordinate-wise law reproducing convolution is complex multiplication (up to one global scale `c`
absorbed into A's scale: `A = sF, g = (1/s)·product`). Hence: fit ⇒ `A`≈DFT and `g`≈multiplication.

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
- `circular_conv(x, h) -> Tensor`: real `(batch, n)` (accept `(n,)` as batch 1). By definition. No FFT.
- `circulant_matrix(h) -> Tensor`: `h` real `(n,)` → real `(n, n)`, `C[t,s] = h[(t-s) mod n]`.

### `src/data.py`
- `make_dataset(n, num, kernel_dist="dense", seed=0) -> dict`
  returns `{"x":(num,n) f32, "h":(num,n) f32, "y":(num,n) f32}`.
  - `x`: standard normal.
  - `h`: `"dense"`=normal; `"sparse"`=`k=max(2,n//8)` random nonzeros (normal), rest 0;
    `"structured"`=normal × random decaying envelope (do NOT describe in frequency terms).
  - `y = circular_conv(x, h)`. No FFT.
- Standard split helper: `train`(dense) / `test`(dense, new seed) / `ood`(sparse and/or structured).

### `src/models.py`
- `class CoordwiseMLP(nn.Module)`: `ℝ^4 → ℝ^2`, 2–3 hidden layers, width ~64, tanh/ReLU. Applied on
  last dim. Shared across all coordinates.
- `class SpectralNeuralModel(nn.Module)`: params `A` complex `(n,n)` (init `eye(n) + 0.01*(randn+i·randn)`),
  plus a `CoordwiseMLP`. `forward(x, h) -> ŷ` per the forward block above (tied A, decoder `inv(A)`).
  Expose `.A` and `.g_phi`.
- (variant) `SpectralNeuralModelUntied`: separate `A` for `u` and `B` for `v`; expose both. Used to
  later check `A ≈ B`.
- **No DFT / no FFT / no hard-coded multiplication / no structural constraint on A here.**

### `src/train.py`
- `train(model, train_ds, test_ds, steps=8000, lr=1e-3, batch=512, log_every=200) -> history`
  Adam; complex MSE; cast inputs to complex inside model; log train/test MSE; save state + history to `results/`.
  This optimization is harder than a hard-coded product — see Training notes.

### `src/baselines.py`
- `class HardProductOracle(nn.Module)`: builds DFT `F[j,k]=exp(-2πi·j·k/n)` internally, uses
  `A=B=F` and hard-coded elementwise product (no MLP). Must reproduce `y` to MSE `< 1e-8` (sanity).
- `class HardProductLearnedBasis(nn.Module)`: learnable `A`, hard-coded elementwise product
  (the previous strong-prior model). Used as an intermediate check that basis-learning works before
  adding the MLP, and as a precision reference.
- `class BilinearTensor(nn.Module)`: `ŷ[t]=Σ_{i,j} W[t,i,j] x[i] h[j]`, complex `W (n,n,n)`.
  Only bilinearity assumed, no basis. Should fit well (convex in W) — reference showing the task is
  learnable with no spectral structure, while Fourier stays hidden in `W`.
- `class MLPControl`, `class NonlinearEncoderControl`: see Experiments (controls).

### `analysis/metrics.py`  (DFT constructed here only)
- `dft_matrix(n) -> Tensor`: `F[j,k]=exp(-2πi·j·k/n)`.
- `diagonalization_residual(A, h) -> float`: `M = A @ circulant_matrix(h).to(complex) @ inv(A)`;
  return `Σ_{i≠j}|M_ij|² / Σ_{ij}|M_ij|²`.
- `mean_diagonalization_residual(A, kernels) -> float`: average over many random `h`.
- `dft_alignment_error(A) -> dict`: row-normalize `A`,`F`; cost `[i,j] = -|⟨A_hat[i],F_hat[j]⟩|`;
  `linear_sum_assignment`; per matched pair optimal complex scale `c=⟨F[j],A[i]⟩/⟨A[i],A[i]⟩`,
  row residual `‖c·A[i]-F[j]‖/‖F[j]‖`. Return `{perm, mean_row_residual, mean_abs_corr}`.
- `unitarity_stats(A) -> dict`: `|A_ij|` mean and `std/mean`; `‖AᴴA−cI‖_F/‖cI‖_F`, `c=tr(AᴴA)/n`.
- **`multiplication_probe(model, mag_range) -> dict`** (key readout for the learned law):
  sample many complex pairs `(u, v)` within the in-distribution magnitude range (estimate range from
  `u=Ax, v=Ah` on data); compute `g_pred = g_phi(u,v)` and the true product `p = u*v`;
  fit a single complex constant `c = ⟨p, g_pred⟩/⟨p, p⟩` (least squares); report
  `rel_residual = ‖g_pred − c·p‖ / ‖c·p‖` and the value `c`. Small residual ⇒ `g_phi` is complex
  multiplication up to one global scale. Also evaluate on `(u,v)` from held-out `x,h` (law generalization).

### `analysis/alignment.py`
Helper(s) for the assignment + per-row complex scaling used above.

### `src/experiments.py`
Runnable; produces `results/` artifacts:
- `n in [16,32,64]`: train `SpectralNeuralModel`; record test MSE, mean diagonalization residual,
  DFT alignment error, unitarity stats, multiplication-probe residual.
- Noise sweep: add Gaussian noise to `y` at several SNRs; record how diagonalization residual and
  probe residual degrade.
- Generalization: train dense → evaluate ood (sparse/structured); record OOD MSE + diagonalization residual.
- Untied variant at n=32: report `‖A−B‖/‖A‖` after gauge alignment (expect small).
- References/controls: `HardProductOracle` MSE; `HardProductLearnedBasis` precision; `BilinearTensor`
  MSE; `NonlinearEncoderControl` (below); `MLPControl` MSE + R² (report Var(y)≈n for context).
- Write `results/mse_table.csv`, `results/diagonalization_error.csv`, `results/multiplication_probe.csv`.

### `visualize.py`
- `|A|` heatmap (expect near-uniform, ≈ 1/√n).
- Alignment figure: matched rows of `A` vs `F` (magnitude & phase) after gauge alignment.
- **Multiplication-probe figure**: scatter of `Re(g_phi(u,v))` vs `Re(c·u·v)` and imag part (expect a line).
- Noise-degradation curve: diagonalization residual & probe residual vs SNR.
- Save PNGs to `results/`.

### `tests/test_sanity.py`
- `circulant_matrix(h) @ x ≈ circular_conv(x, h)`.
- `circular_conv` commutative: `conv(x,h) ≈ conv(h,x)`.
- `HardProductOracle` reproduces `y` to MSE `< 1e-8`.
- `HardProductLearnedBasis` after short training at n=16: test MSE `< 1e-3` (basis-learning works).
- `SpectralNeuralModel` after short training at n=16: test MSE `< 1e-2` and mean diagonalization
  residual `< 0.05` (full discovery pipeline runs).

---

## Recommended defaults & training notes

- `n=32` primary; also `16`, `64`. Train `num=20000` (dense), test/ood `num=4000`.
- `g_phi`: 2–3 hidden layers, width 64, tanh (smooth, helps approximate the product).
- Adam, `lr=1e-3`, minibatch 512, `steps≈8000`. `A⁻¹` via `torch.linalg.inv`.
- **This is harder to optimize than a hard-coded product.** Mitigations:
  - Curriculum: train at `n=8`, warm-start `A`/`g_phi`, then `16`, then `32`.
  - Run ≥3 seeds; keep the best by test MSE.
  - Init `A = eye(n) + 0.01·(randn + i·randn)` (invertible, not DFT-like).
  - If stuck, raise steps / lower lr; do NOT switch in a hard-coded product (that would contaminate
    the discovery).

---

## Acceptance criteria

1. `tests/test_sanity.py` passes.
2. `SpectralNeuralModel` at n=32: test MSE small (e.g. `< 1e-3`), on the order of
   `HardProductLearnedBasis` (the MLP approximation of the product caps precision — acceptable).
3. `mean_diagonalization_residual(A, random kernels)` ≈ 0 (e.g. `< 0.02`).
4. After gauge alignment, `dft_alignment_error.mean_row_residual` small (e.g. `< 0.1`);
   `|A_ij|` near-uniform (`std/mean` small) — unitarity emerged.
5. `multiplication_probe.rel_residual` small (e.g. `< 0.05`) on in-distribution and held-out pairs —
   the multiplication law emerged.
6. OOD (train dense → test sparse/structured) keeps low MSE and low diagonalization residual.
7. `results/` contains the CSV tables and PNG figures listed above.