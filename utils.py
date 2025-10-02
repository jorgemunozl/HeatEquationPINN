import torch
import numpy as np
import matplotlib.pyplot as plt
from config import pinnConfig
import torch.nn.functional as F
import os


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


def append_snapshot(model, snapshots, _, t_sample, evaluation):
    error_a = 0
    for i in t_sample:
        with torch.no_grad():
            time = torch.ones(pinnConfig().error_x_sample) * i
            time = time.unsqueeze(1)
            Y_predict = model(evaluation, time)
            Y_true = heat_function(evaluation, time)
            error_ = error_mape(Y_true, Y_predict)
            error_a += torch.mean(error_)
    snapshots.append((_, error_a, error_a))


def heat_function(x, t):
    a_0 = 1/3
    sum = 0
    for i in range(1, 20):
        exponential = np.exp(-1*pinnConfig().alpha*(2*i*np.pi)**2*t)
        sum += fourier_series(2*i)*np.cos(np.pi*2*i*x)*exponential
    return a_0 + sum


class error():
    def __init__(self, y_true, y_predict):
        self.y_true = y_true
        self.y_predict = y_predict

    def MAPE(self):
        return np.abs(self.y_true-self.y_predict)/np.abs(self.y_predict)

    def MSE(self):
        return self.y_predict

    def error_time(self, model):
        with torch.no_grad():
            x_sample = torch.linspace(0, 1, pinnConfig().error_x_sample)
            x_sample = x_sample.unsqueeze(1)
            t_sample = torch.linspace(0, 1, pinnConfig().error_t_sample)
            time_one = torch.ones((pinnConfig().error_x_sample, 1))
            error_t = []
            for i in t_sample:
                t_eval = time_one * i
                y_predict = model(x_sample, t_eval)
                y_true = heat_function(x_sample.squeeze(), t_eval.squeeze())
                error = MAPE(self.y_true, y_predict)
                error = torch.mean(error)
                error_t.append(error.numpy())
        plt.plot(t_sample.numpy(), error_t)
        plt.show()


class plots():
    def __init__(self, x_sample, t_sample):
        self.x_sample = np.linspace(0, 1, x_sample)
        self.t_sample = np.linspace(0, 1, t_sample)

    def plot_heat_analytic(self, model):
        # From numpy to torch
        base_name = "heat_analytic_"
        os.makedirs("plots/time_fixed", exist_ok=True)
        t_torch = self.t_sample.torch().unsqueeze(1)
        with torch.no_grad():
            x_eval = self.x_sample.torch().unsqueeze(1)
            for i in self.t_sample:
                file_name = base_name+str(round(i, 2))
                t_eval = t_torch * i
                y = model(x_eval, t_eval)
                y_ = heat_function(self.x_sample, i)
                plt.plot(self.x_sample, y_, label=f"True {i}")
                plt.plot(self.x_sample, y, color='')
                plt.legend()
                plt.savefig(file_name)

    def plot_init(self):
        x = np.linspace(0, 1, 100)
        y = initial_condition(x)
        y_ = heat_function(x, 0.0)
        plt.plot(x, y_, label="Heat_0")
        plt.plot(x, y, label="init")
        plt.legend()
        plt.show()

    def three_dimensional(self):
        pass

    def scalar(self):
        pass

    def error_mape_fixed_t(self, model, t: float):
        os.makedirs("error_plots", exist_ok=True)
        file_name = f"error_plots/error_mape_{round(t, 2)}.png"
        with torch.no_grad():
            x_sample = torch.linspace(0, 1, pinnConfig().error_x_sample)
            x_sample = x_sample.unsqueeze(1)
            time_one = torch.ones((pinnConfig().error_x_sample, 1))
            t_eval = time_one * t
            y_predict = model(x_sample, t_eval)
            y_true = heat_function(x_sample, t_eval)
            error_mape_ = error_mape(y_true, y_predict)
        plt.plot(x_sample.numpy(), error_mape_, label="MAPE Error")
        plt.plot(x_sample.numpy(), y_true, label="True")
        plt.plot(x_sample.numpy(), y_predict.numpy(), label="Predicted")
        plt.legend()
        plt.show()
        plt.savefig(file_name)

    def plot_exact(self):
        t = np.linspace(0, 1, self.t_sample)
        x = np.linspace(0, 1, self.x_sample)
        for i in t:
            if i == 0:
                y_ = heat_function(x, 0)
                plt.plot(x, y_, color='red', label=r'$u(x,0)=10(x-x^{2})^{2}$')
                
            x = np.linspace(0, 1, sample)
            y = heat_function(x, i)
            plt.plot(x, y, color='red')  # , label=f'{i.max():.2f}')
        plt.title("Exact Solution")
        plt.legend()
        plt.savefig("neumann/plots/exact_plot.png")

    def plot_predict(self, model):
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

    def plot_both(self, model):
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

    def plot_error_MSE(self, model, t):
        with torch.no_grad():
            x_sample = torch.linspace(0, 1, pinnConfig().error_x_sample)
            x_sample = x_sample.unsqueeze(1)
            time_one = torch.ones((pinnConfig().error_x_sample, 1))
            t_eval = time_one * t
            y_predict = model(x_sample, t_eval)
            y_true = heat_function(x_sample, t_eval)
            error_mape_ = F.mse_loss(y_true, y_predict)
        plt.plot(x_sample.numpy(), error_mape_, label="MAPE")
        plt.plot(x_sample.numpy(), y_true, label="True")
        plt.plot(x_sample.numpy(), y_predict.numpy(), label="Predicted")
        plt.legend()
        plt.show()
        print(torch.mean(error_mape_)*100)
        plt.savefig("neumann/error/erro.png")






def fourier_series(n):
    # n even
    return -480/(np.pi**4*(n**4))








