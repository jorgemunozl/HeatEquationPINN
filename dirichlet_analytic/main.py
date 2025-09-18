import matplotlib.pyplot as plt
import numpy as np


L = 20/3


def function_to_optimize(x):
    return (3*(x**2)-8*x-16)*(np.exp(-0.5*x))

x = np.linspace(0, L, 100)
y = function_to_optimize(x)

plt.plot(x, y)
plt.savefig("dirichlet_analytic/anal.png")
