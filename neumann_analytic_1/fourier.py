import matplotlib.pyplot as plt
import numpy as np
import os


alpha = 0.001


def fourier_series(n):
    # n even
    return -480/(np.pi**4*(n**4))


n_f = 40


def heat_function(x, t: int):
    a_0 = 1/3
    sum = 0
    for i in range(1, n_f):
        exponential = np.exp(-1*alpha*(2*i*np.pi)**2*t)
        sum += fourier_series(2*i)*np.cos(np.pi*2*i*x)*exponential
    return a_0 + sum


def function(x):
    return 10*(x-x**2)**2


points = 1000
L = 1.0
x = np.linspace(0, L, points)
y = function(x)
output_dir = "neumann_analytic_1/heat"
os.makedirs(output_dir, exist_ok=True)


plt.title(r"Heat Equation, $\alpha =10^{-3}$, n=40", fontsize=16)
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"Temperature, $u(x,t)$", fontsize=16)

y = heat_function(x, 0)

plt.plot(x, y, label=r'$u(x,0)=10(x-x^{2})^{2}$')

for i in range(10, 80, 10):
    y_t = heat_function(x, i)
    plt.plot(x, y_t, label=f"t = {i}s")

plt.legend(loc="upper left", fontsize=9)
plt.grid(True)

save_path = os.path.join(output_dir, "heat_equation.png")
plt.savefig(save_path)
print(f"Saved heat plot to {save_path}")
print(f"alpha = {alpha}")
