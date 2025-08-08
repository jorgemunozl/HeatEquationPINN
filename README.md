# Physics-Informed Neural Network (PINN) — Solving Differential Equations

A lightweight project demonstrating how to solve differential equations with a Physics-Informed Neural Network (PINN). PINNs embed the governing physics (differential equations and boundary/initial conditions) directly into the loss function, allowing neural networks to learn solutions that satisfy both data and physics.

---

## Overview
- **Goal**: Learn solutions to ODEs/PDEs by minimizing physics residuals and enforcing boundary/initial conditions.
- **Core idea**: Use automatic differentiation to compute derivatives of the neural network output with respect to inputs, then penalize the differential equation residual.
- **Example**: This README includes a minimal, self-contained PINN for the ODE y' + y = 0 with y(0) = 1.

---

## Installation
- **Python**: 3.9–3.12 recommended
- Create and activate a virtual environment (optional):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install torch numpy matplotlib scipy tqdm
  ```
  If you need a specific PyTorch build (e.g., CUDA), see the official install selector: `https://pytorch.org/get-started/locally/`.

---

## Minimal example: PINN for y' + y = 0, y(0) = 1
This script trains a small MLP to approximate the exact solution y(x) = exp(−x) on x ∈ [0, 1]. You can paste it into a file (e.g., `examples/pinn_ode.py`) and run it after installing the dependencies.

```python
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=1, num_hidden=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Differential equation: y' + y = 0 on [0, 1], with y(0) = 1
# Physics residual r(x) = dy/dx + y

def compute_residual(model, x):
    x.requires_grad_(True)
    y = model(x)
    dy_dx = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]
    residual = dy_dx + y
    return residual

@torch.no_grad()
def exact_solution(x):
    return torch.exp(-x)

def train_pinn(
    num_epochs=5000,
    num_collocation=256,
    lr=1e-3,
    lambda_residual=1.0,
    lambda_bc=10.0,
):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Collocation points in [0, 1]
    x_col = torch.rand((num_collocation, 1), device=device)

    # Boundary condition at x=0: y(0) = 1
    x_bc = torch.zeros((1, 1), device=device)
    y_bc_target = torch.ones((1, 1), device=device)

    pbar = trange(num_epochs, leave=False)
    for _ in pbar:
        optimizer.zero_grad()

        # Residual loss over collocation points
        residual = compute_residual(model, x_col)
        loss_res = torch.mean(residual**2)

        # Boundary loss at x=0
        y0 = model(x_bc)
        loss_bc = torch.mean((y0 - y_bc_target) ** 2)

        loss = lambda_residual * loss_res + lambda_bc * loss_bc
        loss.backward()
        optimizer.step()

        if _ % 100 == 0:
            pbar.set_description(f"loss={loss.item():.4e} res={loss_res.item():.4e} bc={loss_bc.item():.4e}")

    return model

if __name__ == "__main__":
    model = train_pinn()
    model.eval()

    # Evaluate and compare to exact solution
    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
        y_pred = model(x_plot).cpu()
        y_true = exact_solution(x_plot).cpu()

    x_np = x_plot.cpu().numpy().squeeze()
    y_pred_np = y_pred.numpy().squeeze()
    y_true_np = y_true.numpy().squeeze()

    plt.figure(figsize=(6, 4))
    plt.plot(x_np, y_true_np, label="exact: e^{-x}", linewidth=2)
    plt.plot(x_np, y_pred_np, label="PINN", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("PINN solution for y' + y = 0, y(0)=1")
    plt.legend()
    plt.tight_layout()
    plt.show()
```

Run it with:
```bash
python examples/pinn_ode.py
```

---

## How it works (brief)
- **Network**: A small MLP approximates y(x).
- **Autograd**: Compute dy/dx via `torch.autograd.grad` to form the residual.
- **Loss**: Mean squared residual over collocation points + boundary condition loss.
- **Training**: Minimize the combined loss so the solution satisfies the ODE and the BC.

---

## Suggested project structure (flexible)
```
Train/
├─ README.md
├─ examples/
│  └─ pinn_ode.py
├─ src/
│  └─ pinn/
│     ├─ model.py         # MLP and activation choices
│     ├─ equations.py     # Residual builders for ODEs/PDEs
│     ├─ losses.py        # Residual/BC/IC loss terms
│     └─ training.py      # Training/evaluation loops
├─ notebooks/
│  └─ experiments.ipynb
└─ configs/
   └─ ode.yaml            # Hyperparameters and problem definition
```

---

## Extending to other equations
- Replace the residual with your target ODE/PDE (e.g., Poisson, heat, Burgers').
- Add boundary/initial terms as needed.
- Sample collocation points in the spatial/temporal domain of interest.

---

## References
- Raissi, Perdikaris, Karniadakis (2019). Physics-informed neural networks. Journal of Computational Physics. [`https://arxiv.org/abs/1711.10561`](https://arxiv.org/abs/1711.10561)
- PINNs overview and tutorials: [`https://maziarraissi.github.io/`](https://maziarraissi.github.io/)

---

## License
Add your preferred license (e.g., MIT) in a `LICENSE` file. 