import argparse
import itertools
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.benchmark_funcs import get_rosen_minimum
from src.convergence_plot import (
    plot_with_quartiles,
    run_cbe_with_history,
    stats_from_histories,
)

np.random.seed(42)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the output plots.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=5,
        help="Dimensionality of the problem.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["L-BFGS-B", "BFGS"],
        default="L-BFGS-B",
        help="Optimization method to use.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=100,
        help="Number of random seeds to average over.",
    )
    return parser.parse_args()


LB, UB = (0, 3)
OBJ_NAME = "Rosenbrock"
BATCH_SIZES = [1, 2, 5, 10]
MEMORY_SIZES = [10]  # Only for L-BFGS-B, we can vary memory size
INCLUDE_MEMORY_SIZE_TO_LABEL = False
if __name__ == "__main__":
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    x_min = get_rosen_minimum(args.dimension)
    random_initial_points = np.random.uniform(
        LB, UB, size=(math.lcm(*BATCH_SIZES) * args.n_seeds, args.dimension)
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
        random_initial_points = random_initial_points.reshape(
            -1, batch_size, args.dimension
        )
        assert random_initial_points.ndim == 3

        random_seed_histories = []
        for xs0 in random_initial_points:
            res, fvals = run_cbe_with_history(
                xs0, args.method, LB, UB, memory_size=memory_size
            )
            fvals_per_batch = [fval / batch_size for fval in fvals]
            random_seed_histories.append(fvals_per_batch)

        q25, q50, q75 = stats_from_histories(random_seed_histories)
        q25s.append(q25)
        q50s.append(q50)
        q75s.append(q75)
        label = (
            f"$B={batch_size}$, $M={memory_size}$"
            if INCLUDE_MEMORY_SIZE_TO_LABEL
            else f"$B={batch_size}$"
        )
        labels.append(label)

    filename = f"convergence_{OBJ_NAME}_{args.method}_D{args.dimension}"
    filename += f"_UB{UB}_LB{LB}" if args.method == "L-BFGS-B" else ""
    plot_with_quartiles(
        q25s,
        q50s,
        q75s,
        labels,
        ylabel="$\log_{10}$ Objective",
        outpath=os.path.join(args.output_dir, f"{filename}.pdf"),
    )
