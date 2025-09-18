import matplotlib.pyplot as plt
import numpy as np

alpha = 0.1


def fourier_series(n):
    # n even
    return -480/(np.pi**4*(n**4))


def heat_function(x, t):
    a_0 = 1/3
    n = 40
    sum = 0
    for i in range(2, n):
        exponential = np.exp(-1*alpha*(2*i*np.pi)**2*t)
        sum += fourier_series(2*i)*np.cos(np.pi*2*i*x)*exponential
    return a_0 + sum


def function(x):
    return 10*(x-x**2)**2


points = 100
L = 1.0
x = np.linspace(0, L, points)
y = function(x)

plt.plot(x, y)
plt.savefig("neumann_analytic/analitic.png")

t = np.zeros(points)
y = heat_function(x, t)

plt.plot(x, y)
plt.savefig("neumann_analytic/heat.png")
