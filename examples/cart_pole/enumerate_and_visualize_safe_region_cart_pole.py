"""Enumerate probable unsafe regions for CartPole and visualize them on a 2D slice."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

from systems.cart_pole import CartPole
from cersyve import create_mlp
from cersyve.model import create_closed_loop_dynamics_model
from cersyve.enumeration import enumerate_unsafe_regions, BinaryDecisionWrapperCBF


# ============================================================
# Config
# ============================================================
DEVICE = "cpu"

# Slice shown in plots: x(1)=position, x(3)=angle
VIS_DIMS = (0, 2)
FIXED_STATE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Tolerance on hidden dims:
# dim 1 = velocity, dim 3 = angular velocity
# This is the key fix: do not require exact containment of 0.
HIDDEN_TOL = np.array([0.0, 0.35, 0.0, 0.35], dtype=np.float32)

N_POINTS = 260
HORIZON = 100

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# Utilities
# ============================================================
def load_state(model, path, device="cpu"):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).float().eval()
    return model


def build_slice_grid(x_low, x_high, dims=(0, 2), fixed_state=None, n_points=200):
    if fixed_state is None:
        fixed_state = np.zeros(CartPole.x_dim, dtype=np.float32)

    xs = np.linspace(x_low[dims[0]], x_high[dims[0]], n_points, dtype=np.float32)
    ys = np.linspace(x_low[dims[1]], x_high[dims[1]], n_points, dtype=np.float32)
    X1, X2 = np.meshgrid(xs, ys)

    pts = np.repeat(fixed_state[None, :], n_points * n_points, axis=0).astype(np.float32)
    pts[:, dims[0]] = X1.reshape(-1)
    pts[:, dims[1]] = X2.reshape(-1)

    return xs, ys, X1, X2, pts


@torch.no_grad()
def eval_value(V_model, pts, n_points, device="cpu"):
    X = torch.from_numpy(pts.astype(np.float32)).to(device)
    return V_model(X).cpu().numpy().reshape(n_points, n_points)


@torch.no_grad()
def compute_true_safe_region(pts, h_model, f_pi_model, n_points, horizon=100, device="cpu"):
    true_safe = np.ones(pts.shape[0], dtype=bool)
    X_current = torch.from_numpy(pts.T.astype(np.float32)).to(device)  # [4, N]

    for step in range(horizon):
        h_val = h_model(X_current.T).cpu().numpy().flatten()
        true_safe &= (h_val <= 0.0)
        X_current = f_pi_model(X_current)

        if step % 20 == 0:
            print(f"  Step {step}/{horizon}: {true_safe.mean() * 100:.2f}% points remain safe")

    return true_safe.reshape(n_points, n_points)


def project_region_to_slice(region_low, region_high, dims=(0, 2), fixed_state=None, hidden_tol=None):
    """
    Project a 4D box to the 2D slice if it is compatible with the fixed hidden dimensions.
    Compatibility is checked with tolerance on hidden dims.
    """
    if fixed_state is None:
        fixed_state = np.zeros(CartPole.x_dim, dtype=np.float32)
    if hidden_tol is None:
        hidden_tol = np.zeros(CartPole.x_dim, dtype=np.float32)

    hidden_dims = [d for d in range(CartPole.x_dim) if d not in dims]

    for d in hidden_dims:
        lo = region_low[d] - hidden_tol[d]
        hi = region_high[d] + hidden_tol[d]
        if not (lo <= fixed_state[d] <= hi):
            return None

    return (
        float(region_low[dims[0]]),
        float(region_high[dims[0]]),
        float(region_low[dims[1]]),
        float(region_high[dims[1]]),
    )


def load_models(device="cpu"):
    print("Loading CartPole data and models...")

    data_npz = np.load("../../data/cart_pole_data.npz")
    data = {}
    for k in data_npz.keys():
        arr = data_npz[k]
        data[k] = arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr

    x_low = np.asarray(CartPole.x_low, dtype=np.float32)
    x_high = np.asarray(CartPole.x_high, dtype=np.float32)

    # h
    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_model = load_state(h_model, "../../models/cart_pole/h.pth", device)

    # f
    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_model = load_state(f_model, "../../models/cart_pole/f.pth", device)

    # pi
    pi_model = CartPole.create_policy().to(device).float().eval()

    # closed loop
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, x_low, x_high, CartPole.u_dim
    ).to(device).float().eval()

    # pretrained V
    V_pre = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_pre = load_state(V_pre, "../../models/cart_pole/V_pretrain.pth", device)

    print("✓ All models loaded successfully\n")
    return x_low, x_high, h_model, f_pi_model, V_pre


# ============================================================
# Main
# ============================================================
def main():
    x_low, x_high, h_model, f_pi_model, V_pre = load_models(device=DEVICE)

    wrapper_model = BinaryDecisionWrapperCBF(
        V_model=V_pre,
        h_model=h_model,
        f_pi_model=f_pi_model,
    ).to(DEVICE).float().eval()

    input_domain = np.stack([x_low, x_high], axis=1).astype(np.float32)

    print("=" * 70)
    print("Enumerating Probable Unsafe Regions - CartPole")
    print("=" * 70)

    # IMPORTANT:
    # We enumerate unsafe regions in 4D, then project those near the chosen slice.
    unsafe_regions, stats = enumerate_unsafe_regions(
        wrapper_model=wrapper_model,
        input_domain=input_domain,
        n_training_samples=60000,
        n_decision_trees=2500,
        max_depth=14,
        heuristic="gini",
        digits_to_consider=0,
        wilks_filtering=True,
        wilks_samples=3500,
        sample_size_evaluation=10000,
        desired_coverage=70.0,
        enumerate_safe_regions=False,
        incremental_coverage=True,
        verify_safe_regions_with_pgd=False,
        pgd_samples=1000,
        pgd_steps=10,
        pgd_eps=0.1,
        seed=42,
        verbose=True,
        device=DEVICE,
    )

    num_regions = len(unsafe_regions)
    coverage_pct = float(stats.get("coverage", stats.get("coverage_pct", 0.0)))
    error_pct = float(stats.get("error_rate", stats.get("error_pct", 0.0)))

    print("\nEnumeration results")
    print(f"  Regions found: {num_regions}")
    print(f"  Coverage: {coverage_pct:.2f}%")
    print(f"  Error rate: {error_pct:.2f}%")

    # Build 2D slice
    xs, ys, X1, X2, pts = build_slice_grid(
        x_low=x_low,
        x_high=x_high,
        dims=VIS_DIMS,
        fixed_state=FIXED_STATE,
        n_points=N_POINTS,
    )

    # Evaluate pretrained V on the slice
    V_vals = eval_value(V_pre, pts, N_POINTS, DEVICE)

    # True safe region on the slice
    print("\nComputing true safe region (100-step forward simulation on slice)...")
    true_safe_region = compute_true_safe_region(
        pts=pts,
        h_model=h_model,
        f_pi_model=f_pi_model,
        n_points=N_POINTS,
        horizon=HORIZON,
        device=DEVICE,
    )

    # ------------------------------------------------------------
    # Project enumerated unsafe 4D regions onto the slice
    # ------------------------------------------------------------
    projected_rects = []
    enum_unsafe_mask = np.zeros((N_POINTS, N_POINTS), dtype=bool)

    for reg in unsafe_regions:
        region = np.asarray(reg, dtype=np.float32)

        # Expect shape [4,2]
        if region.shape != (CartPole.x_dim, 2):
            continue

        region_low = region[:, 0]
        region_high = region[:, 1]

        proj = project_region_to_slice(
            region_low=region_low,
            region_high=region_high,
            dims=VIS_DIMS,
            fixed_state=FIXED_STATE,
            hidden_tol=HIDDEN_TOL,
        )

        if proj is None:
            continue

        x0, x1, y0, y1 = proj
        projected_rects.append((x0, x1, y0, y1))

        mask = (X1 >= x0) & (X1 <= x1) & (X2 >= y0) & (X2 <= y1)
        enum_unsafe_mask |= mask

    projected_count = len(projected_rects)

    # ------------------------------------------------------------
    # Coverage map logic
    # Compare:
    #   pretrained unsafe prediction   <->   enumerated unsafe projection
    #
    # pretrained_unsafe = V_pre > 0
    # ------------------------------------------------------------
    pretrained_unsafe = (V_vals > 0.0)

    # Labels:
    # 0 -> both safe
    # 1 -> pretrained only
    # 2 -> enumerated only
    # 3 -> both unsafe
    coverage_map = np.zeros((N_POINTS, N_POINTS), dtype=np.int32)
    coverage_map[(~pretrained_unsafe) & (~enum_unsafe_mask)] = 0
    coverage_map[(pretrained_unsafe) & (~enum_unsafe_mask)] = 1
    coverage_map[(~pretrained_unsafe) & (enum_unsafe_mask)] = 2
    coverage_map[(pretrained_unsafe) & (enum_unsafe_mask)] = 3

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    # ---------- Left panel ----------
    ax = axes[0]

    cf = ax.contourf(X1, X2, V_vals, levels=28, cmap="YlGnBu", alpha=0.86)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Pre-trained Value V(x)", fontsize=13)

    ax.contour(
        X1, X2, true_safe_region.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=2.4
    )
    ax.contour(
        X1, X2, V_vals,
        levels=[0.0],
        colors="blue",
        linewidths=2.6
    )

    for (x0, x1, y0, y1) in projected_rects:
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="red",
                edgecolor="red",
                alpha=0.22,
                linewidth=1.0,
            )
        )

    legend_elements = [
        Line2D([0], [0], color="black", lw=2.4, label="True safe region (100-step)"),
        Line2D([0], [0], color="blue", lw=2.6, label="Pre-trained V=0 boundary"),
        Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="red", alpha=0.22,
                  label="Projected enumerated unsafe regions"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11, framealpha=0.95)

    ax.set_title(
        f"CartPole Pre-trained V(x) with Enumerated Unsafe Regions\n"
        f"Projected regions shown: {projected_count}",
        fontsize=17,
        fontweight="bold"
    )
    ax.set_xlabel("x(1) - Cart Position", fontsize=14)
    ax.set_ylabel("x(3) - Pole Angle", fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(float(x_low[0]), float(x_high[0]))
    ax.set_ylim(float(x_low[2]), float(x_high[2]))

    # ---------- Right panel ----------
    ax = axes[1]

    # Explicit, readable colors
    # 0 both safe      -> light gray
    # 1 pretrained only-> blue
    # 2 enumerated only-> red
    # 3 both unsafe    -> dark red
    cmap = ListedColormap([
        "#e6e6e6",  # both safe
        "#0000ff",  # pretrained only
        "#ff2a2a",  # enumerated only
        "#a00000",  # both unsafe
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    im = ax.imshow(
        coverage_map,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    ax.contour(
        X1, X2, true_safe_region.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=2.2
    )

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "Both safe",
        "Pre-trained only",
        "Enumerated only",
        "Both unsafe",
    ])

    ax.set_title(
        f"Coverage Map: Pre-trained vs Enumerated\n"
        f"(Error: {error_pct:.2f}%)",
        fontsize=17,
        fontweight="bold"
    )
    ax.set_xlabel("x(1) - Cart Position", fontsize=14)
    ax.set_ylabel("x(3) - Pole Angle", fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(float(x_low[0]), float(x_high[0]))
    ax.set_ylim(float(x_low[2]), float(x_high[2]))

    fig.suptitle(
        "CartPole: Enumerated Probable Unsafe Regions\n"
        f"Coverage: {coverage_pct:.2f}% | Regions: {num_regions} | "
        f"Slice dims={VIS_DIMS}, fixed_state={FIXED_STATE.tolist()}",
        fontsize=20,
        fontweight="bold"
    )

    out_file = os.path.join(PLOTS_DIR, "enumeration_results_cart_pole.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=320, bbox_inches="tight")
    plt.close(fig)

    print(f"\nVisualization saved to: {out_file}")


if __name__ == "__main__":
    main()