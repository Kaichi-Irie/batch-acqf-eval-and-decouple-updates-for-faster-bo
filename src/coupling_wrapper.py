from typing import Callable

import numpy as np


def couple_f(func: Callable, batch_size: int, dim: int) -> Callable:
    def f(x_flat):
        x = np.reshape(x_flat, (batch_size, dim))
        return np.sum(func(x))

    return f


def couple_grad(grad: Callable, batch_size: int, dim: int) -> Callable:
    def g(x_flat):
        x = np.reshape(x_flat, (batch_size, dim))
        return grad(x).flatten()

    return g
