# %%
import itertools
import math
import os
import statistics
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_funcs import (
    get_rosen_minimum,
    rosenbrock_func,
    rosenbrock_grad,
)
from coupling_wrapper import couple_f, couple_grad

SEED = 42
DIMENSION = 5
MEMORY_SIZE = 10
LB, UB = (0, 3)
BOUNDS = [(LB, UB)] * DIMENSION
METHOD = "L-BFGS-B"  # "L-BFGS-B" or "BFGS"
if METHOD == "BFGS":
    MEMORY_SIZE = None  # BFGSではメモリサイズは不要
    BOUNDS = None

OBJ_NAME = "Rosenbrock"
OUTPUT_DIR = "results/figures/convergence_plot"
np.random.seed(SEED)


def run_cbe_with_history(
    xs0: np.ndarray,
    method: str,
    lb: float,
    ub: float,
    memory_size: int | None = None,
) -> tuple[Any, list[float]]:
    """スタック最適化（合計目的値の履歴を採取）"""
    assert xs0.ndim == 2
    batch_size, dim = xs0.shape
    print(f"Running CBE with B={batch_size}, D={dim}, method={method}")
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
    """
    最適値は、各問題の最適値の和であるので、各バッチでの平均値にする
    """
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


# %%


def pad_histories_to_same_length(histories: list[list[float]]) -> np.ndarray:
    """複数履歴 -> 同じ長さの2次元配列に変換"""
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
    """複数履歴 -> (平均, 標準偏差)"""
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
    """各系列の平均曲線と（任意で）±標準偏差の帯を描画"""
    assert len(means) == len(labels)
    if stds is not None:
        assert len(stds) == len(labels)

    plt.figure(figsize=(8, 6))

    for i, (m, label) in enumerate(zip(means, labels)):
        x = np.arange(len(m))
        (line,) = plt.plot(x, m, label=label)  # 色はmatplotlibに任せる
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

    plt.figure(figsize=(8, 6))

    for i, (q25, q50, q75, label) in enumerate(zip(q25s, q50s, q75s, labels)):
        x = np.arange(len(q50))
        (line,) = plt.plot(x, q50, label=label)  # 色はmatplotlibに任せる
        plt.fill_between(x, q25, q75, alpha=0.1)

    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    # plt.ylim(1e-7, 1e2)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.show()


# %%

x_min = get_rosen_minimum(DIMENSION)
batch_sizes = [1, 2, 5, 10]  # 5, 10]
memory_sizes = [None]  # , 40]  # L-BFGSのメモリサイズ
num_seeds = 300  # 各バッチサイズでのランダム初期点の数（シード数）
random_initial_points = np.random.uniform(
    LB, UB, size=(math.lcm(*batch_sizes) * num_seeds, DIMENSION)
)
# means = []
# stds = []
q25s = []
q50s = []
q75s = []
labels = []

for batch_size, memory_size in itertools.product(batch_sizes, memory_sizes):
    random_initial_points = random_initial_points.reshape(-1, batch_size, DIMENSION)
    assert random_initial_points.ndim == 3

    random_seed_histories = []
    for xs0 in random_initial_points:
        res, hist = run_cbe_with_history(
            xs0, METHOD, LB, UB, memory_size=memory_size
        )
        print(f"Optimization result for B={batch_size}: {res}")
        hist = calculate_average_per_batch(hist, batch_size)
        random_seed_histories.append(hist)

    q25, q50, q75 = stats_from_histories(random_seed_histories)
    q25s.append(q25)
    q50s.append(q50)
    q75s.append(q75)
    label = (
        f"B={batch_size}" if memory_size is None else f"B={batch_size}, M={memory_size}"
    )
    labels.append(label)


# %%
# plot_mean_and_std(
#     means,
#     stds,
#     labels,
#     title,
#     ylabel="Average Objective per Problem (log scale)",
#     outpath=output_filepath,
# )


plot_with_quartiles(
    q25s,
    q50s,
    q75s,
    labels,
    f"Convergence Comparison (median ± IQR) {OBJ_NAME}, {METHOD}, D={DIMENSION}, BD={LB}~{UB}",
    ylabel="Average Objective per Problem (log scale)",
    outpath=f"convergence_plot/convergence_{OBJ_NAME}_{METHOD}_D{DIMENSION}_UB{UB}_LB{LB}_M{10}_quartiles.pdf",
)
