import matplotlib.pyplot as plt
import numpy as np


L = 2


def function_to_optimize(x):
    return 1-(x-1)**2


x = np.linspace(0, L, 100)
y = function_to_optimize(x)

plt.plot(x, y)
plt.savefig("dirichlet_analytic_1/analitic.png")
