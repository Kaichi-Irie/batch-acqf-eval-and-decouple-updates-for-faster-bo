# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as opt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_funcs import (
    get_rosen_minimum,
    rosenbrock_func,
    rosenbrock_grad,
    rosenbrock_hess,
)
from coupling_wrapper import couple_f, couple_grad

# %%
RNG_SEED = 200
BATCH_SIZE = 10
DIMENSION = 5
LB, UB = 0.0, 3.0
METHOD = "BFGS"  # "BFGS"
OBJ_NAME = "Rosenbrock"
OUTPUT_DIR = "hessian_comparison"
np.random.seed(RNG_SEED)


def run_coupled_batch_evaluation(
    x0: np.ndarray, method: str, lb: float, ub: float, batch_size: int, dim: int
):
    f = couple_f(rosenbrock_func, batch_size, dim)
    g = couple_grad(rosenbrock_grad, batch_size, dim)
    res = opt.minimize(
        f,
        x0.flatten(),
        method=method,
        jac=g,
        bounds=[(lb, ub)] * (batch_size * dim) if method == "L-BFGS-B" else None,
    )
    print(f"CBE Optimization: {res.message}")
    return res


def run_sequential_optimization(xs0: np.ndarray, method: str, lb: float, ub: float):
    results = []
    for i in range(xs0.shape[0]):
        res = opt.minimize(
            rosenbrock_func,
            xs0[i],
            method=method,
            jac=rosenbrock_grad,
            bounds=[(lb, ub)] * xs0.shape[1] if method == "L-BFGS-B" else None,
        )
        print(f"Instance {i}: {res.message}")
        results.append(res)
    return results


def hess_and_hess_inv_from_result(res, method: str):
    hess_inv = res.hess_inv.todense() if method == "L-BFGS-B" else res.hess_inv
    hess = np.linalg.inv(hess_inv)
    return hess, hess_inv


def make_block_hess_and_hess_inv(results, method: str):
    parts = []
    parts_inv = []
    for r in results:
        H, Hinv = hess_and_hess_inv_from_result(r, method)
        parts.append(H)
        parts_inv.append(Hinv)
    return linalg.block_diag(*parts), linalg.block_diag(*parts_inv)


def true_hessian_block(points: np.ndarray) -> np.ndarray:
    parts = [rosenbrock_hess(p) for p in points]
    return linalg.block_diag(*parts)


def true_hess_inv_block(points: np.ndarray) -> np.ndarray:
    parts = [np.linalg.inv(rosenbrock_hess(p)) for p in points]
    return linalg.block_diag(*parts)


def compare_hess_and_error(
    hess_true: np.ndarray,
    hess_approx: np.ndarray,
    title: str,
    is_inverse: bool = False,
    error_max: float | None = None,
    filename: str | None = None,
) -> None:
    assert hess_true.shape == hess_approx.shape

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    vmin, vmax = hess_true.min(), hess_true.max()
    vmax += (vmax - vmin) * 0.2
    vmin -= (vmax - vmin) * 0.2

    im0 = axes[0].imshow(hess_true, cmap="viridis", vmin=vmin, vmax=vmax)
    kappa_true = np.linalg.cond(hess_true)
    ax0_title = "True Hessian Inv" if is_inverse else "True Hessian"
    axes[0].set_title(f"{ax0_title}\n($\\kappa$={kappa_true:.2e})")
    axes[0].set_xlabel("Dimensions")
    axes[0].set_ylabel("Dimensions")
    fig.colorbar(im0, ax=axes[0], shrink=0.6)

    im1 = axes[1].imshow(hess_approx, cmap="viridis", vmin=vmin, vmax=vmax)
    kappa_approx = np.linalg.cond(hess_approx)
    axes[1].set_title(f"Approx by {METHOD}\n($\\kappa$={kappa_approx:.2e})")
    axes[1].set_xlabel("Dimensions")
    axes[1].set_ylabel("Dimensions")
    fig.colorbar(im1, ax=axes[1], shrink=0.6)

    if error_max is None:
        error_max = np.abs(hess_true - hess_approx).max()
    im2 = axes[2].imshow(
        np.abs(hess_true - hess_approx), cmap="viridis", vmin=0, vmax=error_max
    )
    error = np.linalg.norm(hess_true - hess_approx, ord="fro")
    axes[2].set_title(f"Absolute Error\n(Frobenius={error:.2e})")
    axes[2].set_xlabel("Dimensions")
    axes[2].set_ylabel("Dimensions")
    fig.colorbar(im2, ax=axes[2], shrink=0.6)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.show()


def compare_hessians(
    hessians: list[np.ndarray],
    labels: list[str],
    title: str,
    true_hess: np.ndarray | None = None,  # used for frobenius norm and color scale
    filename: str | None = None,
) -> None:
    n = len(hessians)
    assert n == len(labels)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if true_hess is not None:
        vmin, vmax = true_hess.min(), true_hess.max()
    else:
        vmin = min(H.min() for H in hessians)
        vmax = max(H.max() for H in hessians)
    vmax += (vmax - vmin) * 0.2
    vmin -= (vmax - vmin) * 0.2

    for i, (H, label) in enumerate(zip(hessians, labels)):
        im = axes[i].imshow(H, cmap="viridis", vmin=vmin, vmax=vmax)
        kappa = np.linalg.cond(H)
        if true_hess is not None:
            fro_err = np.linalg.norm(H - true_hess, ord="fro")
            axes[i].set_title(f"{label}\n($\\kappa$={kappa:.2e}, Fro={fro_err:.2e})")
        else:
            axes[i].set_title(f"{label}\n($\\kappa$={kappa:.2e})")
        axes[i].set_xlabel("Dimensions")
        axes[i].set_ylabel("Dimensions")
        fig.colorbar(im, ax=axes[i], shrink=0.6)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.show()


# %%

xs0 = np.random.uniform(LB, UB, size=(BATCH_SIZE, DIMENSION))

# 真の最適解（検算用）
x_min = get_rosen_minimum(DIMENSION)

# --- Coupled Batched Evaluation ---
res_cbe = run_coupled_batch_evaluation(xs0, METHOD, LB, UB, BATCH_SIZE, DIMENSION)
# assert np.allclose(res_stack.x, np.tile(x_min, BATCH_SIZE), atol=1e-2)

H_cbe, Hinv_cbe = hess_and_hess_inv_from_result(res_cbe, METHOD)

# --- Sequential ---
results_seq = run_sequential_optimization(xs0, METHOD, LB, UB)
# for r in results_seq:
#     assert np.allclose(r.x, x_min, atol=1e-5)
H_seq, Hinv_seq = make_block_hess_and_hess_inv(results_seq, METHOD)

# --- 真のヘッセ（逐次の到達点で評価） ---
pts_seq = np.vstack([r.x for r in results_seq])
H_true_seq = true_hessian_block(pts_seq)
Hinv_true_seq = true_hess_inv_block(pts_seq)

# --- 条件数比較 ---
print("--- Condition Numbers (Hessian and Inverse are the same) ---")
print(f"True Hessian:      {np.linalg.cond(H_true_seq):.2e}")
print(f"Sequential Approx: {np.linalg.cond(H_seq):.2e}")
print(f"CBE Approx:    {np.linalg.cond(H_cbe):.2e}")

# --- 誤差（Frobenius） ---
print("\n--- Frobenius Norm Errors (Hessian) ---")
err_seq = np.linalg.norm(H_seq - H_true_seq, ord="fro")
err_cbe = np.linalg.norm(H_cbe - H_true_seq, ord="fro")
print(f"Sequential Approx Error: {err_seq:.2e}")
print(f"CBE   Approx Error:  {err_cbe:.2e}")

print("\n--- Frobenius Norm Errors (Hessian Inverse) ---")
err_seq_inv = np.linalg.norm(Hinv_seq - Hinv_true_seq, ord="fro")
err_cbe_inv = np.linalg.norm(Hinv_cbe - Hinv_true_seq, ord="fro")
print(f"Sequential Approx Inv Error: {err_seq_inv:.2e}")
print(f"CBE Approx Inv Error:  {err_cbe_inv:.2e}")
# %%

# --- ヒートマップ（真値・逐次近似・スタック近似） ---
# Hessian
compare_hessians(
    [H_true_seq, H_seq, H_cbe],
    ["True Hessian", "Sequential Approx", "CBE Approx"],
    f"Hessian Comparison ({OBJ_NAME}, {METHOD}, B={BATCH_SIZE}, D={DIMENSION})",
    true_hess=H_true_seq,
    filename=f"hessian_comparison_all_{OBJ_NAME}_{METHOD}_B{BATCH_SIZE}_D{DIMENSION}.pdf",
)


# --- 比較図（真値 vs 近似） ---
err_max = max(np.abs(H_true_seq - H_seq).max(), np.abs(H_true_seq - H_cbe).max())
compare_hess_and_error(
    H_true_seq,
    H_cbe,
    f"CBE Approximation ({OBJ_NAME}, {METHOD}, B={BATCH_SIZE}, D={DIMENSION})",
    error_max=err_max,
    filename=f"hessian_comparison_CBE_{OBJ_NAME}_{METHOD}_B{BATCH_SIZE}_D{DIMENSION}.pdf",
)
compare_hess_and_error(
    H_true_seq,
    H_seq,
    f"Sequential Approximation ({OBJ_NAME}, {METHOD}, B={BATCH_SIZE}, D={DIMENSION})",
    error_max=err_max,
    filename=f"hessian_comparison_sequential_{OBJ_NAME}_{METHOD}_B{BATCH_SIZE}_D{DIMENSION}.pdf",
)

# %%
# Hessian Inverse
compare_hessians(
    [Hinv_true_seq, Hinv_seq, Hinv_cbe],
    ["True Hessian Inv", "Sequential Approx Inv", "CBE Approx Inv"],
    f"Hessian Inverse Comparison ({OBJ_NAME}, {METHOD}, B={BATCH_SIZE}, D={DIMENSION})",
    true_hess=Hinv_true_seq,
    filename=f"hessian_inv_comparison_all_{OBJ_NAME}_{METHOD}_B{BATCH_SIZE}_D{DIMENSION}.pdf",
)

# --- 比較図（真値 vs 近似） ---
err_max_inv = max(
    np.abs(Hinv_true_seq - Hinv_seq).max(), np.abs(Hinv_true_seq - Hinv_cbe).max()
)
compare_hess_and_error(
    Hinv_true_seq,
    Hinv_cbe,
    f"CBE Approximation Inv ({OBJ_NAME}, {METHOD}, B={BATCH_SIZE}, D={DIMENSION})",
    error_max=err_max_inv,
    filename=f"hessian_inv_comparison_CBE_{OBJ_NAME}_{METHOD}_B{BATCH_SIZE}_D{DIMENSION}.pdf",
    is_inverse=True,
)
compare_hess_and_error(
    Hinv_true_seq,
    Hinv_seq,
    f"Sequential Approximation Inv ({OBJ_NAME}, {METHOD}, B={BATCH_SIZE}, D={DIMENSION})",
    error_max=err_max_inv,
    filename=f"hessian_inv_comparison_sequential_{OBJ_NAME}_{METHOD}_B{BATCH_SIZE}_D{DIMENSION}.pdf",
    is_inverse=True,
)

# %%
