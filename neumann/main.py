import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

alpha = 0.1


def plot2Heat(model):

    sample = 100
    t_n = np.linspace(0, 1, 10)
    x_ = np.linspace(0, 1, sample)

    with torch.no_grad():
        x = torch.linspace(0, 1, sample).unsqueeze(1)
        for i in t_n:
            fig = plt.figure()
            t = torch.ones((sample)).unsqueeze(1) * i
            y = model(x, t)
            y_ = heat_function(x_, i)
            plt.plot(x_, y_, label=f"True {i}")
            plt.plot(x, y, label="Predicted", color='blue')
            plt.legend()
            plt.savefig(f"neumann/plots/image_{i}.png")
            plt.close(fig)


"""
def plot3Heat(model, t_max=1.0, nx=100, nt=100, save_path=None):
    
    Create a 3D surface of u(x,t) from the PINN `model`.
    - t_max: maximum time to plot (same units used in training)
    - nx, nt: grid resolution for x and t
    - save_path: if given, save figure to this path
    
    # build numpy grids (2D arrays)
    x_np = np.linspace(0, 1, nx)
    t_np = np.linspace(0, t_max, nt)
    X, T = np.meshgrid(x_np, t_np)  # shapes (nt, nx)

    # make flattened torch inputs (N,1)
    X_flat = torch.from_numpy(X.ravel()).float().unsqueeze(1)
    T_flat = torch.from_numpy(T.ravel()).float().unsqueeze(1)

    # evaluate model
    model.eval()
    with torch.no_grad():
        Y_flat = model(X_flat, T_flat).cpu().numpy().ravel()

    # reshape back to grid shape for plotting
    Y = Y_flat.reshape(X.shape)

    # clear/prepare axes
    global ax
    ax.clear()
    surf = ax.plot_surface(X, T, Y, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title(rf'PINN solution, $\alpha={alpha}$')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()
"""


def fourier_series(n):
    # n even
    return -480/(np.pi**4*(n**4))


def heat_function(x, t: int):
    a_0 = 1/3
    sum = 0
    for i in range(1, 40):
        exponential = np.exp(-1*alpha*(2*i*np.pi)**2*t)
        sum += fourier_series(2*i)*np.cos(np.pi*2*i*x)*exponential
    return a_0 + sum


def compute_residual(model, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = model(x, t)
    ones = torch.ones_like(u)
    d_t = torch.autograd.grad(
        u, t, grad_outputs=ones, create_graph=True, retain_graph=True
        )[0]
    d_x = torch.autograd.grad(
        u, x, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]
    d_xx = torch.autograd.grad(
        d_x, x, grad_outputs=torch.ones_like(d_x), create_graph=True,
        retain_graph=True
    )[0]

    return d_t - alpha*d_xx


def initial_condition(x):
    return 10*(x-x**2)**2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layer = [nn.Linear(2, 100), nn.Tanh()]
        for i in range(8):
            layer += [nn.Linear(100, 100), nn.Tanh()]
        layer += [nn.Linear(100, 1), nn.Tanh()]
        self.net = nn.Sequential(*layer)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


def train_pinn(
        num_epochs=5000,
        num_collocation_res=1000,
        num_collocation_ic=500,
        num_collocation_bc=600,
        lr=1e-3,
        lambda_residual=10.0,
        lambda_ic=6.0,
        lambda_bc=5.0
):
    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Residual Collocation
    x_col_res = torch.empty(num_collocation_res, 1).uniform_(0, 1)
    t_col_res = torch.empty(num_collocation_res, 1).uniform_(0, 1)

    # Initial Condition Collocation
    x_col_ic = torch.empty(num_collocation_ic, 1).uniform_(0, 1)
    t_col_ic = torch.zeros((num_collocation_ic, 1))

    # Boundary Condition Collocation
    t_x_bc = torch.empty(num_collocation_bc, 1).uniform_(0, 1)
    x_bc = torch.zeros((num_collocation_bc, 1), requires_grad=True)
    t_l_bc = torch.empty(num_collocation_bc, 1).uniform_(0, 1)
    l_bc = torch.ones((num_collocation_bc, 1), requires_grad=True)

    # Neumann
    ux_0_bc = torch.zeros((num_collocation_bc, 1))
    ux_1_bc = torch.zeros((num_collocation_bc, 1))

    for _ in range(num_epochs):
        optimizer.zero_grad()

        # Residual
        residual = compute_residual(model, x_col_res, t_col_res)
        loss_residual = torch.mean(residual**2)

        # Initial
        model_ic = model(x_col_ic, t_col_ic)
        loss_ic = torch.mean((model_ic-initial_condition(x_col_ic))**2)

        # Boundary
        u_0_bc = model(x_bc, t_x_bc)
        du_0_bc = torch.autograd.grad(
            u_0_bc, x_bc, grad_outputs=torch.ones_like(u_0_bc),
            create_graph=True
        )[0]

        u_l_bc = model(l_bc, t_l_bc)
        du_l_bc = torch.autograd.grad(
            u_l_bc, l_bc, grad_outputs=torch.ones_like(u_l_bc),
            create_graph=True
        )[0]

        loss_0_bc = torch.mean((du_0_bc-ux_0_bc)**2)
        loss_1_bc = torch.mean((du_l_bc-ux_1_bc)**2)
        loss_b = (loss_0_bc + loss_1_bc)
        loss = lambda_residual*loss_residual+lambda_ic*loss_ic+lambda_bc*loss_b
        loss.backward()
        optimizer.step()

        if _ % 200 == 0:
            print(loss)
    save_path = "parametersheat.pth"
    torch.save(
            {'model_state_dict': model.state_dict()}, save_path
        )
    return model


model = NeuralNetwork()
save_path = "parameter_colab_tpu.pth"
loaded = torch.load(save_path)
model.load_state_dict(loaded["model_state_dict"])
model.eval()
plot2Heat(model)
