import numpy as np


def error_fixed_t(y, y_hat):
    return np.abs(y-y_hat)/np.abs(y)


array_1 = np.array([[0.1, 0.2], [0.3, 0.4]])
array_1_hat = np.array([[0.1, 0.4], [0.5, 0.5]])

print(error_fixed_t(array_1, array_1_hat))
