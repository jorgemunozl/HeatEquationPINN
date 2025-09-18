import matplotlib.pyplot as plt
import numpy as np


L = 1


def function_to_optimize(x):
    return (x-x**2)**2*10


x = np.linspace(0, L, 100)
y = function_to_optimize(x)

plt.plot(x, y)
plt.savefig("neumann_analytic_1/analitic.png")
