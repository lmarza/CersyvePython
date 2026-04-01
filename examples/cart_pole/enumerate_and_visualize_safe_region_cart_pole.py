"""Enumerate probable unsafe regions for CartPole and visualize them like Double Integrator."""

import os
import sys
from time import time

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

# Visualized slice: x(1)=position, x(3)=angle
VIS_DIMS = (0, 2)
FIXED_STATE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Hidden dims tolerance:
# dim 1 = velocity, dim 3 = angular velocity
HIDDEN_TOL = np.array([0.0, 0.35, 0.0, 0.35], dtype=np.float32)

N_POINTS = 200
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
    print("Computing true safe region (100-step forward reachability)...")
    true_safe = np.ones(pts.shape[0], dtype=bool)
    X_current = torch.from_numpy(pts.T.astype(np.float32)).to(device)  # [4, N]

    for step in range(horizon):
        X_for_h = X_current.T  # [N, 4]
        h_val = h_model(X_for_h).cpu().numpy().flatten()
        true_safe &= (h_val <= 0.0)
        X_current = f_pi_model(X_current)

    true_safe = true_safe.reshape(n_points, n_points)
    print(f"True safe region computed: {true_safe.mean() * 100:.2f}% of states")
    return true_safe


def project_region_to_slice(region_low, region_high, dims=(0, 2), fixed_state=None, hidden_tol=None):
    """
    Project a 4D box to the 2D slice if compatible with fixed hidden dimensions.
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
    print("\nLoading CartPole data and models...")

    data_npz = np.load("../../data/cart_pole_data.npz")
    data = {}
    for k in data_npz.keys():
        arr = data_npz[k]
        data[k] = arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr

    x_low = np.asarray(CartPole.x_low, dtype=np.float32)
    x_high = np.asarray(CartPole.x_high, dtype=np.float32)

    # constraint model
    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_model = load_state(h_model, "../../models/cart_pole/h.pth", device)

    # dynamics model
    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_model = load_state(f_model, "../../models/cart_pole/f.pth", device)

    # policy
    pi_model = CartPole.create_policy().to(device).float().eval()

    # closed-loop dynamics
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, x_low, x_high, CartPole.u_dim
    ).to(device).float().eval()

    # pretrained value function
    V_pre = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_pre = load_state(V_pre, "../../models/cart_pole/V_pretrain.pth", device)

    print("✓ All models loaded successfully")
    return x_low, x_high, h_model, f_pi_model, V_pre


# ============================================================
# Enumeration
# ============================================================
def enumeration_unsafe_regions_cartpole(V_model, h_model, f_pi_model, x_low, x_high, device):
    print("=" * 80)
    print("Enumerating Probable Unsafe Regions - CartPole")
    print("=" * 80)

    wrapper_model = BinaryDecisionWrapperCBF(
        V_model=V_model,
        h_model=h_model,
        f_pi_model=f_pi_model,
    ).to(device).float().eval()

    input_domain = np.stack([x_low, x_high], axis=1).astype(np.float32)

    start_time = time()
    unsafe_regions, stats = enumerate_unsafe_regions(
        wrapper_model=wrapper_model,
        input_domain=input_domain,
        n_training_samples=200000,
        n_decision_trees=3000,
        max_depth=18,
        heuristic="gini",
        digits_to_consider=0,
        wilks_filtering=False,
        wilks_samples=3500,
        sample_size_evaluation=50000,
        desired_coverage=40.0,
        enumerate_safe_regions=False,
        incremental_coverage=True,
        verify_safe_regions_with_pgd=False,
        pgd_samples=1000,
        pgd_steps=10,
        pgd_eps=0.1,
        seed=42,
        verbose=True,
        device=device,
    )

    elapsed = time() - start_time

    print("\n" + "=" * 80)
    print("Enumeration Results")
    print("=" * 80)
    print(f"Number of unsafe regions found: {len(unsafe_regions)}")
    print(f"Coverage of unsafe samples: {stats.get('coverage', 0.0):.2f}%")
    print(f"Error rate: {stats.get('error_rate', 0.0):.2f}%")
    print(f"Computation time: {elapsed:.2f} seconds - {elapsed/60:.2f} minutes")

    return unsafe_regions, stats, input_domain


# ============================================================
# Visualization
# ============================================================
def visualize_cartpole(h_model, V_model, f_pi_model, unsafe_regions, stats, x_low, x_high):
    print("\n" + "=" * 80)
    print("Creating Visualization")
    print("=" * 80)

    xs, ys, X1, X2, pts = build_slice_grid(
        x_low=x_low,
        x_high=x_high,
        dims=VIS_DIMS,
        fixed_state=FIXED_STATE,
        n_points=N_POINTS,
    )

    # Pretrained value function on slice
    V_pre = eval_value(V_model, pts, N_POINTS, DEVICE)

    # True safe region
    true_safe_region = compute_true_safe_region(
        pts=pts,
        h_model=h_model,
        f_pi_model=f_pi_model,
        n_points=N_POINTS,
        horizon=HORIZON,
        device=DEVICE,
    )

    # Project 4D unsafe regions to 2D slice
    projected_rects = []
    enum_unsafe_mask = np.zeros((N_POINTS, N_POINTS), dtype=bool)

    for reg in unsafe_regions:
        region = np.asarray(reg, dtype=np.float32)
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

        x_min, x_max, y_min, y_max = proj
        projected_rects.append((x_min, x_max, y_min, y_max))

        mask = (X1 >= x_min) & (X1 <= x_max) & (X2 >= y_min) & (X2 <= y_max)
        enum_unsafe_mask |= mask

    # Pretrained unsafe region
    pretrained_unsafe = (V_pre > 0)

    # Coverage map
    # 0: both safe
    # 1: pretrained only
    # 2: enumerated only
    # 3: both unsafe
    coverage_map = np.zeros((N_POINTS, N_POINTS), dtype=np.int32)
    coverage_map[(~pretrained_unsafe) & (~enum_unsafe_mask)] = 0
    coverage_map[(pretrained_unsafe) & (~enum_unsafe_mask)] = 1
    coverage_map[(~pretrained_unsafe) & (enum_unsafe_mask)] = 2
    coverage_map[(pretrained_unsafe) & (enum_unsafe_mask)] = 3

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---------- Plot 1 ----------
    ax = axes[0]
    contour = ax.contourf(X1, X2, V_pre, levels=20, cmap="YlGnBu", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="Value V(x)")

    # true safe boundary
    ax.contour(
        X1, X2, true_safe_region.astype(float),
        levels=[0.5], colors="black", linewidths=2.5, linestyles="-"
    )

    # pretrained boundary
    ax.contour(
        X1, X2, V_pre,
        levels=[0.0], colors="blue", linewidths=2.5, linestyles="-"
    )

    # projected unsafe boxes
    for (x_min, x_max, y_min, y_max) in projected_rects:
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="red",
            facecolor="red",
            alpha=0.35
        )
        ax.add_patch(rect)

    legend_elements = [
        Line2D([0], [0], color="black", lw=2.5, label="True safe region (100-step)"),
        Line2D([0], [0], color="blue", lw=2.5, label="Pre-trained V=0 boundary"),
        Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="red", alpha=0.35,
                  label=f"Projected unsafe regions ({len(projected_rects)})")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.95)

    ax.set_xlabel("Cart Position", fontsize=12)
    ax.set_ylabel("Pole Angle", fontsize=12)
    ax.set_title(
        f"Pre-trained V(x) with Enumerated Unsafe Regions\n({len(projected_rects)} projected regions shown)",
        fontsize=13,
        fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(float(x_low[VIS_DIMS[0]]), float(x_high[VIS_DIMS[0]]))
    ax.set_ylim(float(x_low[VIS_DIMS[1]]), float(x_high[VIS_DIMS[1]]))
    ax.set_aspect("auto")

    # ---------- Plot 2 ----------
    ax = axes[1]

    cmap = ListedColormap(["white", "blue", "red", "darkred"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    img = ax.imshow(
        coverage_map,
        extent=[X1.min(), X1.max(), X2.min(), X2.max()],
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest"
    )

    ax.contour(
        X1, X2, true_safe_region.astype(float),
        levels=[0.5], colors="black", linewidths=2.5, linestyles="-"
    )

    cbar = plt.colorbar(img, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Both safe", "Pre-trained only", "Enumerated only", "Both unsafe"])

    ax.set_xlabel("Cart Position", fontsize=12)
    ax.set_ylabel("Pole Angle", fontsize=12)
    ax.set_title(
        f"Coverage Map: Pre-trained vs Enumerated\n(Error: {stats.get('error_rate', 0.0):.2f}%)",
        fontsize=13,
        fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(float(x_low[VIS_DIMS[0]]), float(x_high[VIS_DIMS[0]]))
    ax.set_ylim(float(x_low[VIS_DIMS[1]]), float(x_high[VIS_DIMS[1]]))
    ax.set_aspect("auto")

    plt.suptitle(
        f"CartPole: Enumerated Probable Unsafe Regions\n"
        f"Coverage: {stats.get('coverage', 0.0):.2f}% | Regions: {len(unsafe_regions)}",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "enumeration_results_cart_pole.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Visualization saved to {out_path}")

    # save regions
    print("\n" + "=" * 80)
    print("Saving Enumerated Regions")
    print("=" * 80)
    regions_save_path = "../../examples/cart_pole/enumerated_unsafe_regions_cart_pole.npz"
    np.savez(
        regions_save_path,
        unsafe_regions=np.array(unsafe_regions, dtype=object),
        stats=stats,
        projected_rects=np.array(projected_rects, dtype=np.float32) if len(projected_rects) > 0 else np.empty((0, 4), dtype=np.float32)
    )
    print(f"✓ Regions saved to: {regions_save_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    x_low, x_high, h_model, f_pi_model, V_pre = load_models(device=DEVICE)
    unsafe_regions, stats, input_domain = enumeration_unsafe_regions_cartpole(
        V_model=V_pre,
        h_model=h_model,
        f_pi_model=f_pi_model,
        x_low=x_low,
        x_high=x_high,
        device=DEVICE,
    )
    visualize_cartpole(
        h_model=h_model,
        V_model=V_pre,
        f_pi_model=f_pi_model,
        unsafe_regions=unsafe_regions,
        stats=stats,
        x_low=x_low,
        x_high=x_high,
    )

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)