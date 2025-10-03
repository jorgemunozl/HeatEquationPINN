import torch
import numpy as np


def fourier_series(n):
    # n even
    return -480/(np.pi**4*(n**4))


def heat_function(x, t):
    a_0 = 1/3
    sum = 0
    for i in range(1, 20):
        exponential = np.exp(-1*1*(2*i*np.pi)**2*t)
        sum += fourier_series(2*i)*np.cos(np.pi*2*i*x)*exponential
    return a_0 + sum


arra = np.array([0.1, 0.5, 0.4])
print(type(arra))
print(type(heat_function(arra, 2)))
