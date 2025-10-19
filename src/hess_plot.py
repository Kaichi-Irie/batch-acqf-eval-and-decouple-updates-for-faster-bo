import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as opt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_funcs import (
    rosenbrock_func,
    rosenbrock_grad,
    rosenbrock_hess,
)
from src.coupling_wrapper import couple_f, couple_grad

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"  # Fonts


def run_coupled_batch_evaluation(
    xs0: np.ndarray, method: str, lb: float, ub: float, batch_size: int, dim: int
):
    f = couple_f(rosenbrock_func, batch_size, dim)
    g = couple_grad(rosenbrock_grad, batch_size, dim)
    res = opt.minimize(
        f,
        xs0.flatten(),
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
    method: str = "L-BFGS-B",
    output_dir: str = "hessian_comparison",
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
    axes[1].set_title(f"Approx by {method}\n($\\kappa$={kappa_approx:.2e})")
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
        plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def relative_error_fro(hess_true: np.ndarray, hess_approx: np.ndarray) -> float:
    """Returns ||H - Ĥ||_F / ||H||_F. Returns np.nan if the true matrix is all zeros."""
    assert hess_true.ndim == 2 and hess_approx.ndim == 2
    assert hess_true.shape == hess_approx.shape
    denom = np.linalg.norm(hess_true, ord="fro")
    if denom == 0:
        return np.nan
    return float(np.linalg.norm(hess_true - hess_approx, ord="fro") / denom)


def plot_hessian_triplet(
    H_true: np.ndarray,
    H_approx_a: np.ndarray,
    H_approx_b: np.ndarray,
    titles=("True", "Approx A", "Approx B"),
    cmap="viridis",
    out_pdf_path="hessian_comparison.pdf",
):
    """
    Compare three matrices (true, approx. 2 types) side-by-side with heatmaps and a common color bar at the right end.
    - Display relative error (Frobenius norm ratio) in the subtitle of the approximation.
    - Minimize margins when saving PDF (bbox_inches='tight', pad_inches=0).
    - Use a common color scale for all 3 plots, with a zero-centered TwoSlopeNorm.
    """
    v_abs = np.nanmax(np.abs(H_true)) * 1.2
    if not np.isfinite(v_abs) or v_abs == 0:
        v_abs = 1.0
    # norm = TwoSlopeNorm(vmin=-v_abs, vcenter=0.0, vmax=v_abs)
    vmin, vmax = H_true.min(), H_true.max()
    vmax += (vmax - vmin) * 0.2
    vmin -= (vmax - vmin) * 0.2
    rel_a = relative_error_fro(H_true, H_approx_a)
    rel_b = relative_error_fro(H_true, H_approx_b)

    plt.rcParams.update(
        {
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7, 3.5),
        constrained_layout=True,
    )

    mats = [H_true, H_approx_a, H_approx_b]
    panel_titles = [
        titles[0],
        f"{titles[1]}\n($e_{{rel}}={rel_a:.2f}$)"
        if np.isfinite(rel_a)
        else f"{titles[1]}\n($e_{{rel}}=$n/a)",
        f"{titles[2]}\n($e_{{rel}}={rel_b:.2f}$)"
        if np.isfinite(rel_b)
        else f"{titles[2]}\n($e_{{rel}}=$n/a)",
    ]

    images = []
    for ax, M, t in zip(axes, mats, panel_titles):
        im = ax.imshow(
            M, cmap=cmap, interpolation="nearest", aspect="equal", vmin=vmin, vmax=vmax
        )
        images.append(im)
        ax.set_title(t, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    cbar = fig.colorbar(
        images[0],
        ax=axes,
        location="right",
        shrink=0.6,
        # pad=0.05,
        fraction=0.05,
        ticks=np.round(np.linspace(vmin + 0.1, vmax - 0.1, num=6), decimals=1),
    )
    cbar.ax.tick_params(labelsize=15)

    fig.savefig(out_pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.show()
    plt.close(fig)
    return out_pdf_path
