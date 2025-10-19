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

RNG_SEED = 4
BATCH_SIZE = 3
DIMENSION = 5
LB, UB = 0.0, 3.0
METHOD = "L-BFGS-B"  # "L-BFGS-B" or "BFGS"
OBJ_NAME = "Rosenbrock"
OUTPUT_DIR = "results_tmp/hessian_comparison"
SUFFIX = f"{OBJ_NAME}_{METHOD}_B{BATCH_SIZE}_D{DIMENSION}_seed{RNG_SEED}"
np.random.seed(RNG_SEED)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xs0 = np.random.uniform(LB, UB, size=(BATCH_SIZE, DIMENSION))
    x_min = get_rosen_minimum(DIMENSION)

    # --- Coupled Batched Evaluation ---
    res_cbe = run_coupled_batch_evaluation(xs0, METHOD, LB, UB, BATCH_SIZE, DIMENSION)
    assert np.allclose(res_cbe.x, np.tile(x_min, BATCH_SIZE), atol=1e-2)

    _, Hinv_cbe = hess_and_hess_inv_from_result(res_cbe, METHOD)

    # --- Sequential ---
    results_seq = run_sequential_optimization(xs0, METHOD, LB, UB)
    _, Hinv_seq = make_block_hess_and_hess_inv(results_seq, METHOD)

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
            OUTPUT_DIR, f"hessian_inv_comparison_triplet_{SUFFIX}.pdf"
        ),
    )
