import math

import numpy as np
import torch
from optuna._gp.acqf import BaseAcquisitionFunc
from optuna._gp.scipy_blas_thread_patch import (
    single_blas_thread_if_scipy_v1_15_or_newer,
)
from optuna.logging import get_logger

from src import batched_lbfgsb as b_opt
from src import iterinfo_global_variables as cfg

_logger = get_logger(__name__)


def _gradient_ascent_batched(
    acqf: BaseAcquisitionFunc,
    initial_params_batched: np.ndarray,
    initial_fvals: np.ndarray,
    continuous_indices: np.ndarray,
    lengthscales: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function optimizes the acquisition function using preconditioning.
    Preconditioning equalizes the variances caused by each parameter and
    speeds up the convergence.

    In Optuna, acquisition functions use Matern 5/2 kernel, which is a function of `x / l`
    where `x` is `normalized_params` and `l` is the corresponding lengthscales.
    Then acquisition functions are a function of `x / l`, i.e. `f(x / l)`.
    As `l` has different values for each param, it makes the function ill-conditioned.
    By transforming `x / l` to `zl / l = z`, the function becomes `f(z)` and has
    equal variances w.r.t. `z`.
    So optimization w.r.t. `z` instead of `x` is the preconditioning here and
    speeds up the convergence.
    As the domain of `x` is [0, 1], that of `z` becomes [0, 1/l].
    """
    if len(continuous_indices) == 0:
        return (
            initial_params_batched,
            initial_fvals,
            np.zeros(len(initial_fvals), dtype=bool),
        )
    normalized_params = initial_params_batched.copy()

    def negative_acqf_with_grad(
        scaled_x: np.ndarray, batch_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Scale back to the original domain, i.e. [0, 1], from [0, 1/s].
        if scaled_x.ndim == 1:
            # NOTE(Kaichi-Irie): When scaled_x is 1D, regard it as a single batch.
            scaled_x = scaled_x[None]
        assert scaled_x.ndim == 2
        normalized_params[np.ix_(batch_indices, continuous_indices)] = (
            scaled_x * lengthscales
        )
        # NOTE(Kaichi-Irie): If fvals.numel() > 1, backward() cannot be computed, so we sum up.
        x_tensor = torch.from_numpy(normalized_params[batch_indices]).requires_grad_(
            True
        )
        neg_fvals = -acqf.eval_acqf(x_tensor)
        neg_fvals.sum().backward()  # type: ignore[no-untyped-call]
        grads = x_tensor.grad.detach().numpy()  # type: ignore[union-attr]
        neg_fvals_ = np.atleast_1d(neg_fvals.detach().numpy())
        # Flip sign because scipy minimizes functions.
        # Let the scaled acqf be g(x) and the acqf be f(sx), then dg/dx = df/dx * s.
        return neg_fvals_, grads[:, continuous_indices] * lengthscales

    with single_blas_thread_if_scipy_v1_15_or_newer():
        scaled_cont_xs_opt, neg_fvals_opt, n_iterations = b_opt.batched_lbfgsb(
            func_and_grad=negative_acqf_with_grad,
            x0_batched=normalized_params[:, continuous_indices] / lengthscales,
            bounds=[(0, 1 / s) for s in lengthscales],
            pgtol=math.sqrt(tol),
            max_iters=200,
        )

    normalized_params[:, continuous_indices] = scaled_cont_xs_opt * lengthscales

    # If any parameter is updated, return the updated parameters and values.
    # Otherwise, return the initial ones.
    fvals_opt = -neg_fvals_opt
    is_updated_batch = (fvals_opt > initial_fvals) & (n_iterations > 0)

    cfg.TOTAL_NITS.append(int(n_iterations.sum()))
    cfg.AVERAGE_NITS.append(
        float(n_iterations[n_iterations > 0].mean())
        if np.any(n_iterations > 0)
        else 0.0
    )
    cfg.MAX_NITS.append(int(n_iterations.max()) if len(n_iterations) > 0 else 0)

    return (
        np.where(is_updated_batch[:, None], normalized_params, initial_params_batched),
        np.where(is_updated_batch, fvals_opt, initial_fvals),
        is_updated_batch,
    )


def local_search_mixed_batched(
    acqf: BaseAcquisitionFunc,
    xs0: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    lengthscales = acqf.length_scales[
        (cont_inds := acqf.search_space.continuous_indices)
    ]
    assert xs0.ndim == 2
    assert len(cont_inds) == xs0.shape[1]
    best_fvals = acqf.eval_acqf_no_grad((best_xs := xs0.copy()))
    best_xs, best_fvals, _ = _gradient_ascent_batched(
        acqf,
        best_xs,
        best_fvals,
        cont_inds,
        lengthscales,
        tol,
    )
    return best_xs, best_fvals


def optimize_acqf_mixed(
    acqf: BaseAcquisitionFunc,
    *,
    warmstart_normalized_params_array: np.ndarray | None = None,
    n_preliminary_samples: int = 2048,
    n_local_search: int = 10,
    tol: float = 1e-4,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.RandomState()

    if warmstart_normalized_params_array is None:
        warmstart_normalized_params_array = np.empty((0, acqf.search_space.dim))

    assert len(warmstart_normalized_params_array) <= n_local_search - 1, (
        "We must choose at least 1 best sampled point + given_initial_xs as start points."
    )

    sampled_xs = acqf.search_space.sample_normalized_params(
        n_preliminary_samples, rng=rng
    )

    # Evaluate all values at initial samples
    f_vals = acqf.eval_acqf_no_grad(sampled_xs)
    assert isinstance(f_vals, np.ndarray)

    max_i = np.argmax(f_vals)
    probs = np.exp(f_vals - f_vals[max_i])
    probs[max_i] = 0.0  # We already picked the best param, so remove it from roulette.
    probs /= probs.sum()
    n_non_zero_probs_improvement = int(np.count_nonzero(probs > 0.0))
    # n_additional_warmstart becomes smaller when study starts to converge.
    n_additional_warmstart = min(
        n_local_search - len(warmstart_normalized_params_array) - 1,
        n_non_zero_probs_improvement,
    )
    if n_additional_warmstart == n_non_zero_probs_improvement:
        _logger.warning(
            "Study already converged, so the number of local search is reduced."
        )
    chosen_idxs = np.array([max_i])
    if n_additional_warmstart > 0:
        additional_idxs = rng.choice(
            len(sampled_xs), size=n_additional_warmstart, replace=False, p=probs
        )
        chosen_idxs = np.append(chosen_idxs, additional_idxs)

    x_warmstarts = np.vstack(
        [sampled_xs[chosen_idxs, :], warmstart_normalized_params_array]
    )
    best_xs, best_fvals = local_search_mixed_batched(acqf, x_warmstarts, tol=tol)
    best_idx = np.argmax(best_fvals).item()
    return best_xs[best_idx], best_fvals[best_idx]
