import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    tanh_x = np.tanh(x)
    return 1 - tanh_x**2
