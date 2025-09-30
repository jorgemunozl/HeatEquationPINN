import torch
import numpy as np
import matplotlib.pyplot as plt
from config import pinnConfig


def plot2Heat(model):
    sample = 100
    t_n = np.linspace(0, 1, 10)
    x_ = np.linspace(0, 1, sample)

    with torch.no_grad():
        x = torch.linspace(0, 1, sample).unsqueeze(1)
        for i in t_n:
            t = torch.ones((sample)).unsqueeze(1) * i
            y = model(x, t)
            y_ = heat_function(x_, i)
            plt.plot(x_, y_, label=f"True {i}")
            plt.plot(x, y, color='blue')
            plt.legend()
            plt.savefig(f"neumann/plots/image_{round(i, 2)}.png")


def error_fixed_t(y, y_hat):
    return np.abs(y-y_hat)/np.abs(y)


def plot_exact():
    sample = 100
    t = np.linspace(0.1, 1, 9)
    x = np.linspace(0, 1, sample)
    y_ = heat_function(x, 0)
    plt.plot(x, y_, color='red', label=r'$u(x,0)=10(x-x^{2})^{2}$')
    for i in t:
        x = np.linspace(0, 1, sample)
        y = heat_function(x, i)
        plt.plot(x, y, color='red')  # , label=f'{i.max():.2f}')
    plt.title("Exact Solution")
    plt.legend()
    plt.savefig("neumann/plots/exact_plot.png")


def plot_predict(model):
    sample = 100
    x_n = np.linspace(0, 1, 10)

    with torch.no_grad():
        x = torch.linspace(0, 1, sample).unsqueeze(1)
        for i in x_n:
            t = torch.ones((sample)).unsqueeze(1) * i
            y = model(x, t)
            plt.plot(x, y, color='blue')
    plt.title('Model Predict')
    plt.savefig("neumann/plots/predicted.png")


def plot_both(model):
    sample = 10000
    t = np.linspace(0, 1, 10)
    x = np.linspace(0, 1, sample)
    y = heat_function(x, 0)
    plt.plot(x, y, color='red', label='Exact')

    for i in t[1:]:
        x = np.linspace(0, 1, sample)
        y = heat_function(x, i)
        plt.plot(x, y, linewidth=1, color='red')
    with torch.no_grad():
        x = torch.linspace(0, 1, sample).unsqueeze(1)
        t_ = torch.zeros((sample)).unsqueeze(1)
        y = model(x, t_)
        plt.plot(x, y, linewidth=1, color='blue', label='Predict')
        for i in t[1:]:
            t = torch.ones((sample)).unsqueeze(1) * i
            y = model(x, t)
            plt.plot(x, y, linewidth=1, color='blue')
    plt.legend()
    plt.savefig("neumann/plots/both.png", dpi=500)


def fourier_series(n):
    # n even
    return -480/(np.pi**4*(n**4))


def heat_function(x, t):
    a_0 = 1/3
    sum = 0
    for i in range(1, 20):
        exponential = np.exp(-1*pinnConfig().alpha*(2*i*np.pi)**2*t)
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

    return d_t - pinnConfig().alpha*d_xx


def initial_condition(x):
    return 10*(x-x**2)**2
