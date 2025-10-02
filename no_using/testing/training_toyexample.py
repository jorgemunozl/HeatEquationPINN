import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=60, out_dim=1, num_hidden=3):
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
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]
    residual = dy_dx + y
    return residual


@torch.no_grad()
def exact_solution(x):
    return torch.exp(-x)


def train_pinn(
    num_epochs=5000,
    num_collocation=512,
    lr=1e-3,
    lambda_residual=1.0,
    lambda_bc=10.0
              ):
    start = time.time()
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Collocation points in [0, 1]
    #x_col = torch.rand((num_collocation, 1), device=device)
    x_col = torch.empty(num_collocation).uniform_(-10.0, 10.0)
    # Boundary condition at x=0: y(0) = 1
    x_col = x_col.view(num_collocation, 1)
    x_bc = torch.zeros((1, 1), device=device)
    y_bc_target = torch.ones((1, 1), device=device)

    for _ in range(num_epochs):
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
            print(f"loss={loss.item():.4e} res={loss_res.item():.4e} bc={loss_bc.item():.4e}")

    save_path = "testing/trained_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': num_epochs
    }, save_path)
    end = time.time()
    print(end-start)
    return model


if __name__ == "__main__":
    model = train_pinn()
    #loaded = torch.load("testing/trained_model.pth",
    #                   map_location=torch.device('cpu'))
    #model = MLP()
    #model.load_state_dict(loaded['model_state_dict'])
    model.eval()

    x = torch.tensor([0.5])
    x.requires_grad_(True)
    y = model(x)
    dy_dx = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]
    residual = dy_dx + y
    print("derivative", dy_dx)
    print("value", y)
    print("residual", residual)

    """
    with torch.no_grad():
        x_plot = torch.linspace(0, 100, 20000, device=device).unsqueeze(1)
        y_pred = model(x_plot).cpu()
        y_true = exact_solution(x_plot).cpu()

    x_np = x_plot.cpu().numpy().squeeze()
    y_pred_np = y_pred.numpy().squeeze()
    y_true_np = y_true.numpy().squeeze()
    plt.figure(figsize=(6, 4))
    plt.plot(x_np, y_true_np, label=r"exact: $e^{-x}$", linewidth=2)
    plt.plot(x_np, y_pred_np, label="PINN", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("PINN solution for y' + y = 0, y(0)=1")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("testing/testing.png")
    """