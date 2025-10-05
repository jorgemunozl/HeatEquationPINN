import torch
import numpy as np
import matplotlib.pyplot as plt
from config import pinnConfig
import os
import matplotlib.animation as animation


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

    return d_t - pinnConfig().alpha * d_xx


def initial_condition(x):
    return 10*(x-x**2)**2 + 3


def heat_function(x, t):
    a_0 = 1/3 + 3
    sum = 0
    for i in range(1, 40):
        exponential = np.exp(-1*pinnConfig().alpha*(2*i*np.pi)**2*t)
        sum += fourier_series(2*i)*np.cos(np.pi*2*i*x)*exponential
    return a_0 + sum


class error():
    def __init__(self, y_true, y_predict):
        self.y_true = y_true
        self.y_predict = y_predict

    def MAPE(self):
        return np.abs(self.y_true-self.y_predict)/np.abs(self.y_true)

    def MSE(self):
        return (self.y_true-self.y_predict)**2


class plots():
    def __init__(self, x_sample=1000, t_sample=10):
        self.x_sample = torch.linspace(0, 1, x_sample).unsqueeze(1)
        self.t_sample = np.linspace(0, 1, t_sample)
        self.time_one = torch.ones((x_sample, 1))
        self.dir = "plots"
        os.makedirs(self.dir, exist_ok=True)

    def heat_comparation(self, model):

        directory = self.dir+"/analytic"
        os.makedirs(directory, exist_ok=True)
        file_name = "first.png"
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

    def three_dimensional(self, model):  # For oct 4
        fig, ax = plt.subplots(figsize=(8, 4), )

        t_sample = torch.linspace(0, 1, len(self.x_sample))
        X, T = torch.meshgrid(self.x_sample, t_sample)

        # make flattened torch inputs (N,1)
        X_flat = X.unsqueeze(1)
        T_flat = T.unsqueeze(1)

        model.eval()
        with torch.no_grad():
            Y_flat = model(X_flat, T_flat).cpu().numpy().ravel()

        Y = Y_flat.reshape(X.shape)

        ax.clear()
        
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title(rf'PINN solution, $\alpha={alpha}$')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()

    def scalar(self):  # For oct 5, use colors to plot.
        pass

    def animation_mape(self, model, epochs=200):
        directory = self.dir+"/animations"
        os.makedirs(directory, exist_ok=True)
        file_name = "animation_mape.mp4"
        file_path = os.path.join(directory, file_name)

        fig, ax = plt.subplots()
        line1, = ax.plot([], [], lw=2, label='MAPE')
        line2, = ax.plot([], [], lw=2, label='True')
        line3, = ax.plot([], [], lw=2, label='Predicted')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

        data1 = []
        data2 = []
        data3 = []

        for t in self.t_sample:
            y_true = heat_function(self.x_sample, t)

            with torch.no_grad():
                y_predict = model(self.x_sample, self.time_one*t)
            E = error(y_true, y_predict)

            data1.append(E.MAPE())
            data2.append(y_true)
            data3.append(y_predict)

        def update(frame):
            y1 = data1[frame]
            y2 = data2[frame]
            y3 = data3[frame]

            line1.set_data(self.x_sample, y1)
            line2.set_data(self.x_sample, y2)
            line3.set_data(self.x_sample, y3)
            return line1, line2, line3,

        animation_ = animation.FuncAnimation(
            fig, update, frames=100
        )
        animation_.save(file_path, writer='ffmpeg')

    def error_mape_fixed_t(self, model, t: float):

        directory = self.dir+"/error"
        os.makedirs(directory, exist_ok=True)
        number = f"{round(t, 2)}"
        file_name = f"error_mape_s_fixed_{number.replace(".", "")}.png"
        file_path = os.path.join(directory, file_name)
        t_eval = self.time_one * t

        with torch.no_grad():
            y_predict = model(self.x_sample, t_eval)

        y_true = heat_function(self.x_sample, t_eval)
        E = error(y_true, y_predict)
        error_mape_ = E.MAPE()

        plt.plot(self.x_sample.numpy(), error_mape_, label="MAPE Error")
        plt.plot(self.x_sample.numpy(), y_true, label="True",
                 linewidth=1)
        plt.plot(self.x_sample.numpy(), y_predict.numpy(),
                 label="Predicted", linewidth=1)
        plt.title(f"Error MAPE for time {t}")
        plt.legend()
        plt.savefig(file_path, dpi=600)

    def error_mse_fixed_t(self, model, t):

        directory = self.dir+"/error"
        os.makedirs(directory, exist_ok=True)
        file_name = f"error_mse_fixed_{int(t*10)}.png"
        file_path = os.path.join(directory, file_name)

        t_eval = self.time_one * t
        y_true = heat_function(self.x_sample, t)

        with torch.no_grad():
            y_predict = model(self.x_sample, t_eval)

        E = error(y_true, y_predict)
        error_mse_ = E.MSE()
        plt.plot(self.x_sample.numpy(), error_mse_, label="MSE")
        plt.plot(self.x_sample.numpy(), y_true, label="True", linewidth=1)
        plt.plot(self.x_sample.numpy(), y_predict.numpy(),
                 label="Predicted", linewidth=1)
        plt.legend()
        plt.savefig(file_path, dpi=700)

    def animate_snapshot(self, model, snap, frame, flag):
        """Plot when using epochs vs error, training(phase),
        flag = True -> save animation, otherwise, save_data"""

        # Plot the true Red
        if flag:
            # Plot the static true.
            # Take the tensor and somehow it plot it and plot it directly
            animation.save()
        else:
            tensor = []
            for t in self.t_sample:
                with torch.no_grad():
                    y_predict = model(self.x_sample, self.time_one*t)
            tensor.append(y_predict)
            snap[frame] = tensor

    def init_con(self):
        y = initial_condition(self.x_sample)
        plt.plot(self.x_sample, y, color='red')
        for i in self.t_sample:
            y_ = heat_function(self.x_sample, self.time_one * i)
            plt.plot(self.x_sample, y_)
        plt.show()
