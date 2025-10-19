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


def run_cbe_with_history(
    xs0: np.ndarray,
    method: str,
    lb: float,
    ub: float,
    memory_size: int | None = None,
) -> tuple[Any, list[float]]:
    assert xs0.ndim == 2
    batch_size, dim = xs0.shape
    history = []
    f = couple_f(rosenbrock_func, batch_size, dim)
    g = couple_grad(rosenbrock_grad, batch_size, dim)

    if method == "L-BFGS-B":
        _, _, res = opt.fmin_l_bfgs_b(
            f,
            xs0.flatten(),
            fprime=g,
            bounds=[(lb, ub)] * (batch_size * dim) if method == "L-BFGS-B" else None,
            callback=lambda xk: history.append(f(xk)),
            m=memory_size if memory_size is not None else 10,
        )
        return res, history
    elif method == "BFGS":
        res = opt.minimize(
            f,
            xs0.flatten(),
            method=method,
            jac=g,
            callback=lambda xk: history.append(f(xk)),
        )
        return res, history
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")


def calculate_average_per_batch(histories: list[float], batch_size: int) -> list[float]:
    return [fval / batch_size for fval in histories]


def plot_convergence(
    curves: list[list[float]],
    labels: list[str],
    title: str,
    stds: list[list[float]] | None = None,
    outpath: str | None = None,
) -> None:
    assert len(curves) == len(labels)
    plt.figure(figsize=(8, 6))
    for curve, label in zip(curves, labels):
        plt.plot(curve, label=label)
    plt.xlabel("Iterations (per batched problems)")
    plt.ylabel("Total Objective Function Value")
    plt.title(title)
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.show()


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
    plt.show()


def plot_with_quartiles(
    q25s: list[np.ndarray],
    q50s: list[np.ndarray],
    q75s: list[np.ndarray],
    labels: list[str],
    title: str,
    ylabel: str = "Objective (log scale)",
    outpath: str | None = None,
) -> None:
    assert len(q25s) == len(q50s) == len(q75s) == len(labels)

    plt.figure(figsize=(5, 2.5))

    for q25, q50, q75, label in zip(q25s, q50s, q75s, labels):
        x = np.arange(len(q50))
        plt.plot(x, q50, label=label)
        plt.fill_between(x, q25, q75, alpha=0.1)

    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    # plt.ylim(1e-7, 1e2)
    plt.title(title, fontsize=16)
    plt.legend()  # fontsize=16)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.show()
