
# HeatEquationPINN — 1D Heat Equation with Physics-Informed Neural Networks (PINN)

This repository contains code, examples and small utilities to train and
evaluate Physics-Informed Neural Networks (PINNs) that solve the one-
dimensional heat equation. The project is exploratory and includes training
scripts, analytic reference solutions, plotting utilities, and a few
experiment checkpoints.

This README gives a concise guide to the repository, how to install the
dependencies, run training and inference, and how to save/load models safely.

## Table of contents

- What this project is
- Quickstart (install + run)
- Training and inference
- Saving / loading models (recommended patterns)
- File layout
- Reproducibility and tips
- Next steps

---

## What this project is

The goal is to approximate the spatio-temporal solution u(x, t) of the
1D heat equation using a neural network whose loss enforces both (1) the PDE
residual at collocation points and (2) boundary / initial conditions.

PINNs are useful when you have physics constraints (PDEs) that you want to
include alongside sparse data. The implementation in this repo is lightweight
and intended for experimentation and learning.

## Quickstart

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .env
source .env/bin/activate
```

2. Install dependencies (CPU example). If you have a CUDA-capable GPU, follow
	 the instructions at https://pytorch.org to install a matching `torch` wheel.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run training (this will create checkpoints under `parameters/` and plots):

```bash
python3 main.py
```

4. Run the inference/visualization example (loads a checkpoint and saves a
	 plot):

```bash
python3 main_dirichlet_inference.py
```

If a script expects a checkpoint with a particular name, open the script and
change the `CHECKPOINT_PATH` variable or pass the path via a CLI option if
available.

## Training and inference

- `main.py` is the main training entrypoint (experiment parameters live in
	`config.py`). It samples collocation points for the PDE residual, computes
	initial/boundary losses, and optimizes the network.
- `main_dirichlet_inference.py` is a small example showing how to load a
	checkpoint (state dict), compute predictions on a grid, compute a simple
	MSE against an analytic expression, and save the figure `prove.png`.

Read the top of these files for quick customization (checkpoints names,
plot outputs, and domain ranges).

## Saving and loading models — recommended patterns

Best practice: save the `state_dict()` together with a small `config` dict and
any optimizer/training metadata. This makes it trivial to reconstruct the
network at load time.

Example (save checkpoint):

```python
torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'config': model_config,   # e.g. dict with layer sizes / activation
		'epoch': epoch,
		'loss': loss_value,
}, 'parameters/checkpoint.pth')
```

Load safely for inference:

```python
ckpt = torch.load('parameters/checkpoint.pth', map_location='cpu')
cfg = ckpt['config']
model = NeuralNetwork(**cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

If you only need inference and prefer a tiny file, save only the `state_dict`
with `torch.save(model.state_dict(), 'model_state.pth')`, but remember to
recreate the same model architecture in code before calling
`load_state_dict`.

Avoid using `torch.save(model, ...)` for long-term storage — it pickles the
whole object and can break across PyTorch versions or refactors.

## File layout (high level)

- `main.py` — training script
- `main_dirichlet_inference.py` — inference + plotting example
- `neumann_analytic_1/` — analytic solutions and plotting helpers for tests
- `parameters/` — saved checkpoints used in experiments
- `plots/`, `plots_newton/`, `neumann_analytic_1/heat/` — output images
- `src/` — project source (e.g. `config.py`, `utils.py`)
- `requirements.txt` — minimal Python dependencies

## Reproducibility and tips

- Save a small `config` dictionary with each checkpoint (hyperparameters,
	architecture). This makes experiments reproducible and checkpoints self-
	descriptive.
- Use `map_location=torch.device('cpu')` when loading GPU-saved checkpoints on
	a CPU-only machine.
- Fix random seeds in scripts when you want deterministic runs.

## Next steps (suggestions)

- Add CLI flags to `main.py` (`argparse`) to control training vs. inference,
	checkpoint paths and training hyperparameters.
- Add a `requirements-lock.txt` (pinned versions) or `environment.yml` for
	reproducible environments.
- Add unit tests for core utilities (residual computation, analytic solver
	comparisons) so changes are safer to refactor.

---

If you want, I can now:
- Add a `requirements-lock.txt` with exact working versions.
- Wire `main.py` to accept `--checkpoint` and `--eval-only` CLI args.
- Provide a short `example_train_and_eval.py` showing train -> save -> load ->
	infer using the exact `NeuralNetwork` class in this repo.

Tell me which of these you want next and I'll implement it.

