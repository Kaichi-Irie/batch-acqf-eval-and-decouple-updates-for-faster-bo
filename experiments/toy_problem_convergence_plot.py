import itertools
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.benchmark_funcs import get_rosen_minimum
from src.convergence_plot import (
    calculate_average_per_batch,
    plot_with_quartiles,
    run_cbe_with_history,
    stats_from_histories,
)

np.random.seed(42)

DIMENSION = 5
MEMORY_SIZE = 10
LB, UB = (0, 3)
BOUNDS = [(LB, UB)] * DIMENSION
METHOD = "L-BFGS-B"  # "L-BFGS-B" or "BFGS"
if METHOD == "BFGS":
    MEMORY_SIZE = None
    BOUNDS = None

OBJ_NAME = "Rosenbrock"
OUTPUT_DIR = "results_tmp/convergence_plot"
N_SEEDS = 100

BATCH_SIZES = [1, 2, 5, 10]
MEMORY_SIZES = [None]  # Only for L-BFGS-B, we can vary memory size

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x_min = get_rosen_minimum(DIMENSION)
    random_initial_points = np.random.uniform(
        LB, UB, size=(math.lcm(*BATCH_SIZES) * N_SEEDS, DIMENSION)
    )
    # means = []
    # stds = []
    q25s = []
    q50s = []
    q75s = []
    labels = []

    for batch_size, memory_size in itertools.product(BATCH_SIZES, MEMORY_SIZES):
        print(
            f"Running experiments with batch size={batch_size}, memory size={memory_size}"
        )
        random_initial_points = random_initial_points.reshape(-1, batch_size, DIMENSION)
        assert random_initial_points.ndim == 3

        random_seed_histories = []
        for xs0 in random_initial_points:
            res, hist = run_cbe_with_history(
                xs0, METHOD, LB, UB, memory_size=memory_size
            )
            hist = calculate_average_per_batch(hist, batch_size)
            random_seed_histories.append(hist)

        q25, q50, q75 = stats_from_histories(random_seed_histories)
        q25s.append(q25)
        q50s.append(q50)
        q75s.append(q75)
        label = (
            f"$B={batch_size}$"
            if memory_size is None
            else f"$B={batch_size}$, $M={memory_size}$"
        )
        labels.append(label)

    filename = f"convergence_{OBJ_NAME}_{METHOD}_D{DIMENSION}"
    filename += f"_UB{UB}_LB{LB}_M{MEMORY_SIZE}" if METHOD == "L-BFGS-B" else ""
    plot_with_quartiles(
        q25s,
        q50s,
        q75s,
        labels,
        "Convergence Comparison (median ± IQR)",
        ylabel="Objective Value",
        outpath=os.path.join(OUTPUT_DIR, f"{filename}.pdf"),
    )
