import torch
import numpy as np
import matplotlib.pyplot as plt
from config import pinnConfig
import os


def fourier_series(n):
    # n even
    return -480/(np.pi**4*(n**4))


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
        return (self.y_true-self.y_predict)**2


class plots():
    def __init__(self, x_sample=1000, t_sample=1000):
        self.x_sample = torch.linspace(0, 1, x_sample).unsqueeze(1)
        self.t_sample = np.linspace(0, 1, t_sample)
        self.dir = "plots"
        os.makedirs(self.dir, exist_ok=True)

    def heat_comparation(self, model):

        directory = self.dir+"/analytic"
        os.makedirs(directory, exist_ok=True)
        file_name = "comparation.png"
        os.makedirs("plots/time_fixed", exist_ok=True)
        file_path = os.path.join(directory, file_name)

        with torch.no_grad():
            t_torch = torch.ones((1000, 1), dtype=torch.float)
            for i in self.t_sample:
                t_eval = t_torch * i
                y = model(self.x_sample, t_eval)
                y_ = heat_function(self.x_sample, i)
                if i == 0:
                    plt.plot(self.x_sample, y_, color='red',
                             label='True', linewidth=1)
                    plt.plot(self.x_sample, y, color='blue',
                             label='Predicted', linewidth=1)
                plt.plot(self.x_sample, y, color='blue', linewidth=1)
                plt.plot(self.x_sample, y_, color='red', linewidth=1)
        plt.title("Comparation")
        plt.legend()
        plt.savefig(file_path, dpi=600)

    def time_vs_error(self, model):

        directory = self.dir+"/error"
        os.makedirs(directory, exist_ok=True)
        file_name = "time_vs_error.png"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)

        with torch.no_grad():
            time_one = torch.ones((len(self.x_sample), 1))
            error_t = np.zeros((len(self.t_sample)))
            cont = 0
            for i in self.t_sample:
                t_eval = time_one * i
                y_predict = model(self.x_sample, t_eval)
                y_true = heat_function(self.x_sample.squeeze(),
                                       t_eval.squeeze())
                E = error(y_true, y_predict)
                error_ = E.MAPE()
                error_ = torch.mean(error_)
                error_t[cont] = error_.numpy()
                cont += 1
        plt.plot(self.t_sample, error_t)
        plt.savefig(file_path, dpi=600)

    def three_dimensional(self):
        pass

    def scalar(self):
        pass

    def error_mape_fixed_t(self, model, t: float):
        directory = self.dir+"/error"
        os.makedirs(directory, exist_ok=True)
        file_name = f"error_mape_fixed_{int(t*10)}.png"
        file_path = os.path.join(directory, file_name)
        with torch.no_grad():
            x_sample = torch.linspace(0, 1, pinnConfig().error_x_sample)
            x_sample = x_sample.unsqueeze(1)

            time_one = torch.ones((pinnConfig().error_x_sample, 1))
            t_eval = time_one * t

            y_predict = model(x_sample, t_eval)
            y_true = heat_function(x_sample, t_eval)

            E = error(y_true, y_predict)
            error_mape_ = E.MAPE()

        plt.plot(x_sample.numpy(), error_mape_, label="MAPE Error")
        plt.plot(x_sample.numpy(), y_true, label="True")
        plt.plot(x_sample.numpy(), y_predict.numpy(), label="Predicted")
        plt.title(f"Error MAPE for time {t}")
        plt.legend()
        plt.savefig(file_path, dpi=500)

    def error_mse_fixed_t(self, model, t):

        directory = self.dir+"/error"
        os.makedirs(directory, exist_ok=True)
        file_name = f"error_mse_fixed_{int(t*10)}.png"
        file_path = os.path.join(directory, file_name)

        with torch.no_grad():
            time_one = torch.ones((len(self.x_sample), 1))
            t_eval = time_one * t
            y_predict = model(self.x_sample, t_eval)
            y_true = heat_function(self.x_sample, t)
            E = error(y_true, y_predict)
            error_mse_ = E.MSE()
        plt.plot(self.x_sample.numpy(), error_mse_, label="MSE")
        plt.plot(self.x_sample.numpy(), y_true, label="True")
        plt.plot(self.x_sample.numpy(), y_predict.numpy(), label="Predicted")
        plt.legend()
        plt.savefig(file_path, dpi=600)
