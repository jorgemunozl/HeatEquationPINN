import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    snapshot_every=500,
):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Collocation points in [0, 1]
    x_col = torch.empty(num_collocation, 1).uniform_(0, 1).to(device)

    # Boundary condition at x=0: y(0) = 1
    x_bc = torch.zeros((1, 1), device=device)
    y_bc_target = torch.ones((1, 1), device=device)

    snapshots = []

    for epoch in range(num_epochs + 1):
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

        if epoch % 100 == 0:
            print(f"epoch={epoch} loss={loss.item():.4e} res={loss_res.item():.4e} bc={loss_bc.item():.4e}")

        if epoch % snapshot_every == 0:
            with torch.no_grad():
                x_plot = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
                y_pred = model(x_plot).cpu().numpy().squeeze()
                snapshots.append((epoch, x_plot.cpu().numpy().squeeze(), y_pred))

    return model, snapshots

if __name__ == "__main__":
    model, snapshots = train_pinn()

    # Exact solution
    x_ref = snapshots[0][1]
    y_true = np.exp(-x_ref)

    # Animation
    fig, ax = plt.subplots(figsize=(6, 4))
    line_pred, = ax.plot([], [], "--", label="PINN")
    line_true, = ax.plot(x_ref, y_true, "k", linewidth=2, label="exact: e^{-x}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.legend()

    def update(frame):
        epoch, x, y_pred = snapshots[frame]
        line_pred.set_data(x, y_pred)
        ax.set_title(f"PINN solution, epoch {epoch}")
        return line_pred,

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True, repeat=False)
    plt.tight_layout()
    plt.show()
