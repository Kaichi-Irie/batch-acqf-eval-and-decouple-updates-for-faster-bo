from collections.abc import Callable

import numpy as np
import scipy.optimize as so
from greenlet import greenlet


def batched_lbfgsb(
    func_and_grad: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    x0_batched: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    m: int = 10,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    max_evals: int = 15000,
    max_iters: int = 15000,
    max_line_search: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert x0_batched.ndim == 2, (
        f"The shape of x0 must be (batch_size, dim), but got {x0_batched.shape}."
    )
    batch_size = len(x0_batched)
    xs_opt = np.empty_like(x0_batched)
    fvals_opt = np.empty(batch_size, dtype=float)
    n_iterations = np.empty(batch_size, dtype=int)
    is_remaining_batch = np.ones(batch_size, dtype=bool)

    def run(i: int) -> None:
        def _func_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            fval, grad = greenlet.getcurrent().parent.switch(x)
            return float(fval), grad.copy()

        x_opt, fval_opt, info = so.fmin_l_bfgs_b(
            func=_func_and_grad,
            x0=x0_batched[i],
            bounds=bounds,
            m=m,
            factr=factr,
            pgtol=pgtol,
            maxfun=max_evals,
            maxiter=max_iters,
            maxls=max_line_search,
        )
        xs_opt[i] = x_opt
        fvals_opt[i] = fval_opt
        n_iterations[i] = info["nit"]
        is_remaining_batch[i] = False

    greenlets = [greenlet(run) for _ in range(batch_size)]
    x_batched = [gl.switch(i) for i, gl in enumerate(greenlets)]

    while np.any(is_remaining_batch):
        remaining_batch_indices = np.where(is_remaining_batch)[0]
        fvals, grads = func_and_grad(
            np.asarray(x_batched), np.asarray(remaining_batch_indices)
        )

        x_batched = []
        next_greenlets = []
        for i, gl in enumerate(greenlets):
            x = gl.switch((fvals[i], grads[i]))
            if x is None:
                continue
            x_batched.append(x)
            next_greenlets.append(gl)
        greenlets = next_greenlets
    return xs_opt, fvals_opt, n_iterations
