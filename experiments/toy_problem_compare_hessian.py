import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_funcs import get_rosen_minimum
from src.hess_plot import (
    hess_and_hess_inv_from_result,
    make_block_hess_and_hess_inv,
    plot_hessian_triplet,
    run_coupled_batch_evaluation,
    run_sequential_optimization,
    true_hess_inv_block,
)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the output plots.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["L-BFGS-B", "BFGS"],
        default="L-BFGS-B",
        help="Optimization method to use.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=5,
        help="Dimensionality of the problem.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Batch size for the optimization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4,
        help="Random seed for initialization.",
    )
    return parser.parse_args()


LB, UB = 0.0, 3.0
OBJ_NAME = "Rosenbrock"

if __name__ == "__main__":
    args = parse()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = (
        f"{OBJ_NAME}_{args.method}_B{args.batch_size}_D{args.dimension}_seed{args.seed}"
    )
    xs0 = np.random.uniform(LB, UB, size=(args.batch_size, args.dimension))
    x_min = get_rosen_minimum(args.dimension)

    # --- Coupled Batched Evaluation ---
    res_cbe = run_coupled_batch_evaluation(
        xs0, args.method, LB, UB, args.batch_size, args.dimension
    )

    _, Hinv_cbe = hess_and_hess_inv_from_result(res_cbe, args.method)

    # --- Sequential ---
    results_seq = run_sequential_optimization(xs0, args.method, LB, UB)
    _, Hinv_seq = make_block_hess_and_hess_inv(results_seq, args.method)

    pts_seq = np.vstack([r.x for r in results_seq])
    Hinv_true_seq = true_hess_inv_block(pts_seq)

    print("\n--- Frobenius Norm Errors (Inverse Hessian ) ---")
    relative_err_seq_inv = np.linalg.norm(
        Hinv_seq - Hinv_true_seq, ord="fro"
    ) / np.linalg.norm(Hinv_true_seq, ord="fro")
    relative_err_cbe_inv = np.linalg.norm(
        Hinv_cbe - Hinv_true_seq, ord="fro"
    ) / np.linalg.norm(Hinv_true_seq, ord="fro")
    print(f"Sequential Approx Inv Error: {relative_err_seq_inv:.2e}")
    print(f"CBE Approx Inv Error:  {relative_err_cbe_inv:.2e}")
    plot_hessian_triplet(
        Hinv_true_seq,
        Hinv_seq,
        Hinv_cbe,
        titles=("(a) True", "(b) Seq. Opt.", "(c) C-BE"),
        out_pdf_path=os.path.join(
            args.output_dir, f"hessian_inv_comparison_triplet_{suffix}.pdf"
        ),
    )
