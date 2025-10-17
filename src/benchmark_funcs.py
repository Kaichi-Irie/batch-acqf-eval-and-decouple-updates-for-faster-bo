import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess


def rosenbrock_func(x: np.ndarray) -> np.ndarray:
    """
    Rosenbrock Function.
    Input: (N,D) shape, Output: (N,) shape
    """
    assert x.ndim in (1, 2), "Input must be 1D or 2D array."
    return np.array([rosen(xi) for xi in x]) if x.ndim == 2 else rosen(x)


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    assert x.ndim in (1, 2), "Input must be 1D or 2D array."
    if x.ndim == 1:
        return rosen_der(x)
    return np.vstack([rosen_der(xi) for xi in x])


def rosenbrock_hess(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    assert x.ndim in (1, 2), "Input must be 1D or 2D array."
    if x.ndim == 1:
        return rosen_hess(x)  # type: ignore
    return np.array([rosen_hess(xi) for xi in x])  # type: ignore


def styblinski_tang_func(x: np.ndarray) -> np.ndarray:
    """
    Styblinski-Tang Function.
    Input: (N,D) shape, Output: (N,) shape
    """
    return np.sum(x**4 - 16 * x**2 + 5 * x, axis=-1) / 2


def styblinski_tang_grad(x: np.ndarray) -> np.ndarray:
    return (4 * x**3 - 32 * x + 5) / 2


def styblinski_tang_hessian(x: np.ndarray) -> np.ndarray:
    return np.diag((12 * x**2 - 32) / 2)


def dixon_price_func1D(x: np.ndarray) -> float:
    assert x.ndim == 1, "x should be a 1D array."
    term1 = (x[0] - 1) ** 2
    term2 = sum(
        (float(i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2) for i in range(1, len(x))
    )
    return term1 + term2


def dixon_price_func(x: np.ndarray) -> np.ndarray:
    """
    Dixon-Price Function.
    Input: (N,D) shape, Output: (N,) shape"""
    if x.ndim == 1:
        x = x[None, :]
    return np.array([dixon_price_func1D(xi) for xi in x]).squeeze()


def dixon_price_grad1D(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 1, "x should be a 1D array."
    n = len(x)
    grad = np.zeros(n)

    grad[0] = 2 * (x[0] - 1) - 4 * (2 * x[1] ** 2 - x[0])

    for i in range(1, n - 1):
        coeff = float(i + 1)
        grad[i] = 8 * coeff * (2 * x[i] ** 2 - x[i - 1]) * x[i] - 2 * (
            float(i + 2) * (2 * x[i + 1] ** 2 - x[i])
        )

    coeff = float(n)
    grad[n - 1] = 8 * coeff * (2 * x[n - 1] ** 2 - x[n - 2]) * x[n - 1]

    return grad


def dixon_price_grad(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[None, :]
    return np.array([dixon_price_grad1D(xi) for xi in x]).squeeze()


def dixon_price_hess(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 1, "x should be a 1D array."
    n = len(x)
    hess = np.zeros((n, n))

    hess[0, 0] = 2.0

    for i in range(1, n):
        coeff = float(i + 1)

        hess[i, i] += coeff * (8 * (6 * x[i] ** 2 - x[i - 1]))
        hess[i - 1, i - 1] += coeff * 2
        hess[i, i - 1] += coeff * (-8 * x[i])
        hess[i - 1, i] += coeff * (-8 * x[i])

    return hess


def get_dixon_price_minimum(dimension: int) -> np.ndarray:
    if dimension < 1:
        raise ValueError("Dimension n must be greater than or equal to 1.")

    x_min = np.zeros(dimension)
    for i in range(1, dimension + 1):
        exponent = (2**i - 2) / 2**i
        x_min[i - 1] = 2 ** (-exponent)

    return x_min


def get_rosen_minimum(dimension: int) -> np.ndarray:
    return np.ones(dimension)
