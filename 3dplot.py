import numpy as np
import torch
import matplotlib.pyplot as plt

fig , ax = plt.subplots(figsize=(8,4), )


def plot3Heat(model, t_max=1.0, nx=100, nt=100, save_path=None):
    """
    Create a 3D surface of u(x,t) from the PINN `model`.
    - t_max: maximum time to plot (same units used in training)
    - nx, nt: grid resolution for x and t
    - save_path: if given, save figure to this path
    """

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
