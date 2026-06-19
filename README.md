# Fourier Discovery

This AI-for-Math project learns circular convolution from random examples while discovering both:

- a complex linear basis `A` where convolution becomes coordinate independent;
- a shared coordinate-wise neural interaction `g_phi` that should learn complex multiplication up to scale.

The discovery model is:

```text
u = A x
v = A h
z[k] = g_phi(Re u[k], Im u[k], Re v[k], Im v[k])
y_hat = A^{-1} z
```

`A` is an unconstrained `torch.complex64` matrix. `g_phi` is a generic MLP shared across coordinates. The direct data generator uses circular convolution from the definition in `src/convolution.py`; no FFT or built-in convolution is used in `src/data.py`, `src/convolution.py`, `src/models.py`, or `src/train.py`.

The full experiment uses a two-phase optimization schedule for the discovery model:

- learn the basis with an off-diagonal penalty on `A C_h A^{-1}`, which directly enforces the allowed coordinate-independence assumption without constructing the DFT or hard-coding multiplication;
- fix the unidentifiable per-row scale gauge by keeping each row of `A` at unit norm, which makes the shared MLP's learned law comparable across coordinates;
- freeze `A` and train the generic coordinate MLP from convolution examples so the per-coordinate law emerges.

## AI Assistance

This project was developed with assistance from OpenAI Codex, a coding agent based on GPT-5.

## Environment

Create a local Python environment:

```bash
conda create -y -n fourier-discovery python=3.11
conda activate fourier-discovery
pip install -r requirements.txt
```

The code automatically uses CUDA when `torch.cuda.is_available()` is true and otherwise runs on CPU.

## Quick Checks

```bash
PYTHONPATH=. pytest -q
```

Run a small smoke experiment:

```bash
PYTHONPATH=. python -m src.experiments --mode smoke --results-dir /tmp/fourier-discovery-smoke
PYTHONPATH=. python visualize.py --results-dir /tmp/fourier-discovery-smoke
```

Run the full experiment suite:

```bash
PYTHONPATH=. python -m src.experiments --mode full
PYTHONPATH=. python visualize.py --checkpoint results/spectral_neural_n32_full_model.pt
```

Results are written as CSV tables, PyTorch checkpoints, JSON histories, and PNG figures.

## Files

- `src/convolution.py`: direct circular convolution and circulant matrices.
- `src/data.py`: deterministic synthetic datasets.
- `src/models.py`: coordinate-wise MLP discovery models.
- `src/train.py`: complex MSE training loop.
- `src/baselines.py`: DFT oracle, hard-product references, bilinear and MLP controls.
- `analysis/metrics.py`: diagonalization, DFT alignment, unitarity, and multiplication-probe metrics.
- `src/experiments.py`: size, noise, OOD, untied, reference, and control experiments.
- `visualize.py`: figures for learned transforms and multiplication probes.
