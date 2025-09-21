import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

LAMBDA = 0.5


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, 100), nn.Tanh()]
        for _ in range(4):
            layers += [nn.Linear(100, 100), nn.Tanh()]
        layers += [nn.Linear(100, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def computeResidual(model, x):
    x.requires_grad_(True)
    y = model(x)
    dy_dx = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]
    d2y_dx2 = torch.autograd.grad(
        dy_dx, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]
    residual = d2y_dx2 + LAMBDA*y
    return residual


def exact(x):
    w = np.sqrt(LAMBDA)
    c = 3*np.cos(3*w)-4/3*np.sin(2*w)
    c = c/(np.cos(w))
    d = 4/w*np.cos(2*w)+3*np.sin(3*w)
    d = d/(np.cos(w))

    return c*np.cos(w*x) + d*np.sin(w*x)


def train_pinn(
        num_epochs=5000,
        num_collocation=512,
        lr=1e-3,
        lambda_residual=5.0,
        lambda_ic_1=1.0,
        lambda_ic_2=2.0,
        ):

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_col = torch.empty(num_collocation, 1).uniform_(-2.0, 2.0)
    x_ic = torch.tensor([2.0], requires_grad=True)
    y_ic_target = torch.tensor([3.0])
    x_ic1 = torch.tensor([3.0], requires_grad=True)
    y_x_target = torch.tensor([4.0])

    for _ in range(num_epochs):
        optimizer.zero_grad()
        residual = computeResidual(model, x_col)
        loss_res = torch.mean(residual**2)

        y_ic = model(x_ic)
        loss_ic = torch.mean((y_ic-y_ic_target)**2)

        y_ic2 = model(x_ic1)
        dy_x_ic = torch.autograd.grad(
            y_ic2, x_ic1, grad_outputs=torch.ones_like(y_ic2),
            create_graph=True, retain_graph=True
        )[0]
        loss_ic2 = torch.mean((dy_x_ic - y_x_target)**2)
        loss = lambda_residual * loss_res + lambda_ic_1 * loss_ic + lambda_ic_2 * loss_ic2
        loss.backward()
        optimizer.step()

        if _ % 100 == 0:
            print(loss)
    save_path = "mas/parameters.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': num_epochs,
    }, save_path)

    return model


if __name__ == "__main6__":

    loaded = torch.load("mas/parameters.pth")
    model = Net()
    model.load_state_dict(loaded['model_state_dict'])
    model.eval()
    frontier = 6
    x = torch.linspace(-1*frontier, frontier, 1000).unsqueeze(1)
    y = model(x)
    print("a")
    x_n = x.detach().numpy().squeeze()
    y_n = y.detach().numpy().squeeze()
    plt.plot(x_n, y_n)
    y = exact(x_n)
    plt.plot(x_n, y)
    plt.show()
