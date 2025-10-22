import os
import statistics
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_funcs import rosenbrock_func, rosenbrock_grad
from src.coupling_wrapper import couple_f, couple_grad

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"  # Fonts


# Okabe–Ito colorblind-safe palette (8 colors)
OKABE_ITO = [
    "#D55E00",  # vermilion
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#F0E442",  # yellow
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#000000",  # black
]


def run_cbe_with_history(
    xs0: np.ndarray, method: str, lb: float, ub: float, memory_size: int = 10
) -> tuple[Any, list[float]]:
    assert xs0.ndim == 2
    batch_size, dim = xs0.shape
    fvals_history = []
    f = couple_f(rosenbrock_func, batch_size, dim)
    g = couple_grad(rosenbrock_grad, batch_size, dim)

    if method == "L-BFGS-B":
        _, _, res = opt.fmin_l_bfgs_b(
            f,
            xs0.flatten(),
            fprime=g,
            bounds=[(lb, ub)] * (batch_size * dim) if method == "L-BFGS-B" else None,
            callback=lambda xk: fvals_history.append(f(xk)),
            m=memory_size,
            pgtol=1e-12,
            maxiter=100_000,
            maxfun=150_000,
            factr=10,
        )
        return res, fvals_history
    elif method == "BFGS":
        res = opt.fmin_bfgs(
            f,
            xs0.flatten(),
            fprime=g,
            callback=lambda xk: fvals_history.append(f(xk)),
            gtol=1e-12,
            maxiter=300_000,
        )
        return res, fvals_history
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")


def plot_convergence(
    curves: list[list[float]],
    labels: list[str],
    stds: list[list[float]] | None = None,
    outpath: str | None = None,
) -> None:
    assert len(curves) == len(labels)
    plt.figure(figsize=(8, 6))
    for curve, label in zip(curves, labels):
        plt.plot(curve, label=label)
    plt.xlabel("Iterations (per batched problems)")
    plt.ylabel("Total Objective Function Value")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    # plt.show()


def pad_histories_to_same_length(histories: list[list[float]]) -> np.ndarray:
    median_length = statistics.median(len(h) for h in histories)
    median_length = int(median_length)
    median_length = int(median_length)

    padded = []
    for h in histories:
        if len(h) < median_length:
            h = h + [h[-1]] * (median_length - len(h))
        padded.append(h[:median_length])
    return np.array(padded, dtype=float)


def stats_from_histories(
    histories: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = pad_histories_to_same_length(histories)
    mean = np.mean(H, axis=0)
    std = np.std(H, axis=0, ddof=0)
    q25 = np.percentile(H, 25, axis=0)
    q50 = np.percentile(H, 50, axis=0)  # median
    q75 = np.percentile(H, 75, axis=0)
    return q25, q50, q75


def plot_mean_and_std(
    means: list[np.ndarray],
    stds: list[np.ndarray] | None,
    labels: list[str],
    title: str,
    ylabel: str = "Objective (log scale)",
    outpath: str | None = None,
    plot_mean_only: bool = False,
) -> None:
    assert len(means) == len(labels)
    if stds is not None:
        assert len(stds) == len(labels)

    plt.figure(figsize=(8, 6))

    for i, (m, label) in enumerate(zip(means, labels)):
        x = np.arange(len(m))
        (line,) = plt.plot(x, m, label=label)
        if (not plot_mean_only) and stds is not None:
            s = stds[i]
            plt.fill_between(x, m - s, m + s, alpha=0.2)

    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    # plt.show()


def plot_with_quartiles(
    q25s: list[np.ndarray],
    q50s: list[np.ndarray],
    q75s: list[np.ndarray],
    labels: list[str],
    ylabel: str = "Objective (log scale)",
    outpath: str | None = None,
) -> None:
    assert len(q25s) == len(q50s) == len(q75s) == len(labels)

    plt.figure(figsize=(5.5, 2.5))

    for i, q25, q50, q75, label in zip(range(len(q25s)), q25s, q50s, q75s, labels):
        x = np.arange(len(q50))
        color = OKABE_ITO[i % len(OKABE_ITO)]
        plt.plot(x, q50, label=label, color=color)
        plt.fill_between(x, q25, q75, alpha=0.1, facecolor=color)
        if len(x) > 225:
            plt.xlim(0, 225)

    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    # plt.show()
