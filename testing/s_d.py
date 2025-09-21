import torch
import torch.nn as nn
import matplotlib.pyplot as plt


device = 'cpu'


class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=100, out_dim=1, num_hidden=6):
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
    num_collocation=5120,
    lr=1e-3,
    lambda_residual=1.0,
    lambda_bc=10.0
              ):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Collocation points in [0, 1]
    x_col = torch.empty(num_collocation, 1, device=device).uniform_(-2.0, 2.0)
    # Boundary condition at x=0: y(0) = 1
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
            print(f"loss={loss.item():.4e} res={loss_res.item():.4e} bc={loss_bc.item():.4e}",_)

    save_path = "trained_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': num_epochs
    }, save_path)
    return model


if __name__ == "__main__":
    loaded = torch.load("testing/trained_model_colab.pth",
                        map_location=torch.device('cpu'))
    model = MLP()
    model.load_state_dict(loaded['model_state_dict'])
    model.eval()

    x = torch.linspace(-4, 4, 10000, device='cpu').unsqueeze(1)
    x.requires_grad_(True)
    y = model(x)
    dy_dx = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

    d2y_dx2 = torch.autograd.grad(dy_dx, x,
                                  grad_outputs=torch.ones_like(dy_dx),
                                  create_graph=True, retain_graph=True)[0]

    x_n = x.detach().numpy().squeeze()
    y_numpy = y.detach().numpy().squeeze()
    dy_dx = dy_dx.detach().numpy().squeeze()
    d2y_dx2 = d2y_dx2.detach().numpy().squeeze()
    plt.plot(x_n, dy_dx)
    plt.plot(x_n, d2y_dx2)
    plt.show()
