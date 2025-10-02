import matplotlib.pyplot as plt
import numpy as np


L = 4*np.pi


def function_to_optimize(x):
    return np.arctan(0.1*x)*np.sin(x)


x = np.linspace(0, L, 100)
y = function_to_optimize(x)

plt.plot(x, y)
plt.savefig("neumann_analytic/analitic.png")
