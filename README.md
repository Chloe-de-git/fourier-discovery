# Fourier Discovery

This project learns a complex linear transform that turns circular convolution into coordinate-wise multiplication:

```text
y_hat = A^{-1}((A x) * (B h))
```

The learned matrices are unconstrained `torch.complex64` parameters. The ground-truth circular convolution used for data generation is implemented directly from the definition in `src/convolution.py`.

## AI Assistance

This project was developed with assistance from OpenAI Codex, a coding agent based on GPT-5.

## Environment

Use a project-local conda environment under `/data/ziling`:

```bash
/data/ziling/miniconda3/bin/conda create -y -n fourier-discovery python=3.11
/data/ziling/miniconda3/envs/fourier-discovery/bin/pip install -r requirements.txt
```

The code automatically uses CUDA when `torch.cuda.is_available()` is true and otherwise runs on CPU.

## Quick Checks

```bash
PYTHONPATH=. /data/ziling/miniconda3/envs/fourier-discovery/bin/pytest -q
```

Run a small smoke experiment:

```bash
PYTHONPATH=. /data/ziling/miniconda3/envs/fourier-discovery/bin/python -m src.experiments --mode smoke
```

Run the full experiment suite:

```bash
PYTHONPATH=. /data/ziling/miniconda3/envs/fourier-discovery/bin/python -m src.experiments --mode full
PYTHONPATH=. /data/ziling/miniconda3/envs/fourier-discovery/bin/python visualize.py --checkpoint results/untied_n32_model.pt
```

Results are written to `results/` as CSV tables, PyTorch checkpoints, JSON histories, and PNG figures.

## Files

- `src/convolution.py`: direct circular convolution and circulant matrices.
- `src/data.py`: deterministic synthetic datasets.
- `src/models.py`: tied and untied complex spectral models.
- `src/train.py`: complex MSE training loop.
- `src/baselines.py`: DFT oracle plus real MLP/CNN baselines.
- `analysis/metrics.py`: diagonalization, DFT alignment, spectrum, and unitarity metrics.
- `src/experiments.py`: size, noise, OOD, tied-vs-untied, and baseline experiments.
- `visualize.py`: figures for learned transforms and spectra.
