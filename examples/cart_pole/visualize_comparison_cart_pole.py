"""Visualize CartPole pretrained vs finetuned value functions with counterexamples."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import matplotlib.pyplot as plt

from systems.cart_pole import CartPole
from cersyve import create_mlp
from cersyve.utils import uniform
from cersyve.model import create_closed_loop_dynamics_model


# ============================================================
# Config
# ============================================================
DEVICE = "cpu"

# CartPole state:
# 0 = cart position
# 1 = cart velocity
# 2 = pole angle
# 3 = pole angular velocity
VIS_DIMS = (0, 2)  # position vs angle
FIXED_STATE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

N_POINTS = 260
HORIZON = 100
COUNTEREXAMPLE_SAMPLES = 28

# Filter points that are reasonably close to the displayed slice
VEL_THR = 0.35
ANGVEL_THR = 0.35

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
    """Build a 2D slice grid inside the 4D CartPole state space."""
    if fixed_state is None:
        fixed_state = np.zeros(CartPole.x_dim, dtype=np.float32)

    xs = np.linspace(x_low[dims[0]], x_high[dims[0]], n_points, dtype=np.float32)
    ys = np.linspace(x_low[dims[1]], x_high[dims[1]], n_points, dtype=np.float32)
    X1, X2 = np.meshgrid(xs, ys)

    pts = np.repeat(fixed_state[None, :], n_points * n_points, axis=0).astype(np.float32)
    pts[:, dims[0]] = X1.reshape(-1)
    pts[:, dims[1]] = X2.reshape(-1)

    return xs, ys, X1, X2, pts


def project_states(states_4d, dims=(0, 2)):
    """Project [4, N] states to [2, N] on the chosen slice dims."""
    if states_4d.shape[1] == 0:
        return np.empty((2, 0), dtype=np.float32)
    return states_4d[dims, :]


def compute_tfr(V_model, h_model, f_pi_model, n_samples=100_000, horizon=100, device="cpu",
                return_missed=False):
    """Compute True Feasible Rate on the full 4D state space."""
    V_model.eval()

    X = uniform(CartPole.x_low, CartPole.x_high, n_samples).astype(np.float32)  # [4, N]
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        V = V_model(X_t.T).T.cpu().numpy()[0, :]  # [N]

        truly_feasible = np.ones(n_samples, dtype=bool)
        X_current = X_t.clone()  # [4, N]

        for _ in range(horizon):
            h_current = h_model(X_current.T).T.cpu().numpy()[0, :]
            truly_feasible &= (h_current <= 0)
            X_current = f_pi_model(X_current)

    n_truly_feasible = truly_feasible.sum()
    identified = (V <= 0) & truly_feasible
    n_identified = identified.sum()
    missed = truly_feasible & (V > 0)

    tfr = n_identified / max(n_truly_feasible, 1)
    result = {
        "tfr": tfr,
        "n_truly_feasible": int(n_truly_feasible),
        "n_identified": int(n_identified),
        "n_missed": int(missed.sum()),
        "truly_feasible_rate": n_truly_feasible / n_samples,
    }
    if return_missed:
        result["X_missed"] = X[:, missed]
        result["V_missed"] = V[missed]
    return result


@torch.no_grad()
def eval_value(V_model, pts, n_points, device="cpu"):
    X_torch = torch.from_numpy(pts.astype(np.float32)).to(device)  # [N, 4]
    return V_model(X_torch).cpu().numpy().reshape(n_points, n_points)


@torch.no_grad()
def eval_constraint(h_model, pts, n_points, device="cpu"):
    X_torch = torch.from_numpy(pts.astype(np.float32)).to(device)  # [N, 4]
    return h_model(X_torch).cpu().numpy().reshape(n_points, n_points)


@torch.no_grad()
def eval_next_value(V_model, pts, f_pi_model, n_points, device="cpu"):
    X_for_dynamics = torch.from_numpy(pts.T.astype(np.float32)).to(device)  # [4, N]
    X_next = f_pi_model(X_for_dynamics).T.cpu().numpy()  # [N, 4]
    X_next_torch = torch.from_numpy(X_next.astype(np.float32)).to(device)
    return V_model(X_next_torch).cpu().numpy().reshape(n_points, n_points)


@torch.no_grad()
def compute_true_safe_region(pts, h_model, f_pi_model, n_points, horizon=100, device="cpu"):
    """Compute true safe region on the chosen slice by 100-step simulation."""
    true_safe_region = np.ones(pts.shape[0], dtype=bool)
    X_current = torch.from_numpy(pts.T.astype(np.float32)).to(device)  # [4, N]

    for step in range(horizon):
        h_current = h_model(X_current.T).cpu().numpy().flatten()
        true_safe_region &= (h_current <= 0)
        X_current = f_pi_model(X_current)

        if step % 20 == 0:
            print(f"  Step {step}/{horizon}: {true_safe_region.mean() * 100:.2f}% states remain safe")

    return true_safe_region.reshape(n_points, n_points)


def load_shared_models(device="cpu"):
    """Load data + h + f + pi + closed-loop + V_pre."""
    print("Loading CartPole data and shared models...")

    data_path = "../../data/cart_pole_data.npz"
    data_npz = np.load(data_path)
    data = {}
    for key in data_npz.keys():
        arr = data_npz[key]
        data[key] = arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr

    x_low = np.asarray(CartPole.x_low, dtype=np.float32)
    x_high = np.asarray(CartPole.x_high, dtype=np.float32)

    # Constraint model
    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_model = load_state(h_model, "../../models/cart_pole/h.pth", device)

    # Dynamics model
    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_model = load_state(f_model, "../../models/cart_pole/f.pth", device)

    # Policy
    pi_model = CartPole.create_policy().to(device).float().eval()

    # Closed-loop dynamics
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, x_low, x_high, CartPole.u_dim
    ).to(device).float().eval()

    # Pretrained value
    V_pretrain = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_pretrain = load_state(V_pretrain, "../../models/cart_pole/V_pretrain.pth", device)

    print("✓ Shared models loaded\n")
    return x_low, x_high, h_model, f_pi_model, V_pretrain


def filter_and_order_counterexamples(states_4d, n_keep=28, vel_thr=0.35, angvel_thr=0.35):
    """
    Make CartPole counterexamples look more like DI:
    - keep points close to the displayed slice in hidden dims
    - if too many remain, keep the best ones
    - sort by x-position for a visually linear trend
    """
    if states_4d.shape[1] == 0:
        return states_4d

    x = states_4d.copy()  # [4, N]

    # hidden dims are 1=velocity and 3=angular velocity
    mask = (np.abs(x[1, :]) <= vel_thr) & (np.abs(x[3, :]) <= angvel_thr)

    filtered = x[:, mask]

    # fallback if filtering is too strict
    if filtered.shape[1] < max(6, n_keep // 3):
        filtered = x

    # score = closeness to the slice in hidden dims
    score = np.abs(filtered[1, :]) + np.abs(filtered[3, :])

    if filtered.shape[1] > n_keep:
        order = np.argsort(score)
        filtered = filtered[:, order[:n_keep]]

    # sort by x-position to imitate the linear progression seen in DI
    order_x = np.argsort(filtered[0, :])
    filtered = filtered[:, order_x]

    return filtered


def sample_counterexamples(mask, X_points_4d, n_samples=30):
    """Sample points from a boolean mask."""
    indices = np.where(mask.flatten())[0]
    if len(indices) == 0:
        return np.empty((CartPole.x_dim, 0), dtype=np.float32)

    if len(indices) > n_samples * 6:
        rng = np.random.default_rng(42)
        indices = rng.choice(indices, n_samples * 6, replace=False)

    return X_points_4d[indices].T  # [4, K]


def visualize_comparison(
    fine_tune_model_path="../../models/cart_pole/V_finetune_milp.pth",
    name_file="plots/comparison_visualization_milp.png",
    title_suffix="MILP"
):
    device = DEVICE

    x_low, x_high, h_model, f_pi_model, V_pretrain = load_shared_models(device=device)

    # Load finetuned value function
    V_finetune = create_mlp(CartPole.x_dim, 1, [64, 64])
    print(f"Loading finetuned value function from {fine_tune_model_path}...")
    V_finetune = load_state(V_finetune, fine_tune_model_path, device)

    # Build slice grid
    _, _, X1, X2, pts = build_slice_grid(
        x_low=x_low,
        x_high=x_high,
        dims=VIS_DIMS,
        fixed_state=FIXED_STATE,
        n_points=N_POINTS,
    )

    # Compute values
    print("Computing value functions and constraints on the CartPole slice...")
    with torch.no_grad():
        H = eval_constraint(h_model, pts, N_POINTS, device)
        V_pre = eval_value(V_pretrain, pts, N_POINTS, device)
        V_fine = eval_value(V_finetune, pts, N_POINTS, device)
        V_next_pre = eval_next_value(V_pretrain, pts, f_pi_model, N_POINTS, device)
        V_next_fine = eval_next_value(V_finetune, pts, f_pi_model, N_POINTS, device)

    # Compute true safe region
    print("Computing true safe region (100-step forward reachability)...")
    true_safe_region = compute_true_safe_region(
        pts=pts,
        h_model=h_model,
        f_pi_model=f_pi_model,
        n_points=N_POINTS,
        horizon=HORIZON,
        device=device,
    )
    print(f"True safe region computed: {true_safe_region.mean() * 100:.2f}% of slice states\n")

    # ============================================================
    # Counterexamples -- USE PRETRAIN, like DI
    # ============================================================
    constraint_counter_pre = (V_pre <= 0) & (H > 0)
    constraint_counter_fine = (V_fine <= 0) & (H > 0)

    invariance_counter_pre = (V_pre <= 0) & (V_next_pre > 0)
    invariance_counter_fine = (V_fine <= 0) & (V_next_fine > 0)

    all_pre_counter_mask = constraint_counter_pre | invariance_counter_pre
    all_pre_counter_raw = sample_counterexamples(
        all_pre_counter_mask,
        X_points_4d=pts,
        n_samples=COUNTEREXAMPLE_SAMPLES,
    )  # [4, K_raw]

    # Filter and order for a cleaner, DI-like figure
    all_pre_counter = filter_and_order_counterexamples(
        all_pre_counter_raw,
        n_keep=COUNTEREXAMPLE_SAMPLES,
        vel_thr=VEL_THR,
        angvel_thr=ANGVEL_THR,
    )

    all_pre_counter_2d = project_states(all_pre_counter, dims=VIS_DIMS)

    # Next states of those counterexamples
    if all_pre_counter.shape[1] > 0:
        with torch.no_grad():
            counter_torch = torch.from_numpy(all_pre_counter.astype(np.float32)).to(device)
            counter_next = f_pi_model(counter_torch).cpu().numpy()  # [4, K]
        counter_next_2d = project_states(counter_next, dims=VIS_DIMS)
    else:
        counter_next_2d = np.empty((2, 0), dtype=np.float32)

    # TFR
    trf_fine = compute_tfr(V_finetune, h_model, f_pi_model, n_samples=100_000, horizon=100)["tfr"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(X1, X2, V_fine, levels=26, cmap="YlGnBu", alpha=0.82)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Fine-tuned Value V(x)", fontsize=13)

    ax.contour(
        X1, X2, true_safe_region.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=2.5,
        linestyles="-"
    )

    ax.contour(
        X1, X2, V_pre,
        levels=[0],
        colors="orange",
        linewidths=2.2,
        linestyles="-"
    )

    ax.contour(
        X1, X2, V_fine,
        levels=[0],
        colors="blue",
        linewidths=2.8,
        linestyles="-"
    )

    # Counterexamples from PRETRAIN
    if all_pre_counter_2d.shape[1] > 0:
        ax.scatter(
            all_pre_counter_2d[0, :],
            all_pre_counter_2d[1, :],
            c="red",
            s=58,
            alpha=0.60,
            edgecolors="darkred",
            linewidth=1.4,
            marker="o",
            zorder=5,
        )

    # Next states
    if counter_next_2d.shape[1] > 0:
        ax.scatter(
            counter_next_2d[0, :],
            counter_next_2d[1, :],
            c="darkred",
            s=58,
            alpha=0.82,
            edgecolors="black",
            linewidth=1.3,
            marker="o",
            zorder=5,
        )

        for i in range(all_pre_counter_2d.shape[1]):
            ax.plot(
                [all_pre_counter_2d[0, i], counter_next_2d[0, i]],
                [all_pre_counter_2d[1, i], counter_next_2d[1, i]],
                color="darkred",
                alpha=0.40,
                linewidth=1.0,
                zorder=4,
            )

    ax.set_xlabel("x(1) - Position", fontsize=14)
    ax.set_ylabel("x(3) - Pole Angle", fontsize=14)
    ax.set_title(
        f"CartPole: Pre-trained vs Fine-tuned Value Functions ({title_suffix})",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    ax.grid(True, alpha=0.28)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=2.5, label="True region"),
        Line2D([0], [0], color="orange", lw=2.2, label="Pre-trained region"),
        Line2D([0], [0], color="blue", lw=2.8, label="Fine-Tuned region"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=9, label="Counterexample", markeredgecolor="darkred", markeredgewidth=1.3),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkred",
               markersize=9, label="Next state", markeredgecolor="black", markeredgewidth=1.3),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.95, edgecolor="black")

    plt.tight_layout()
    plt.savefig(name_file, dpi=320, bbox_inches="tight")
    print(f"\nVisualization saved to: {name_file}")

    # Stats
    print("\n" + "=" * 70)
    print("Value Function Comparison Statistics")
    print("=" * 70)

    print("\nPre-trained Model:")
    print(f"  Safe region size (V <= 0): {(V_pre <= 0).mean() * 100:.2f}%")
    print(f"  Constraint counterexamples: {constraint_counter_pre.sum()} points ({constraint_counter_pre.mean() * 100:.3f}%)")
    print(f"  Invariance counterexamples: {invariance_counter_pre.sum()} points ({invariance_counter_pre.mean() * 100:.3f}%)")
    print(f"  Total counterexamples: {(constraint_counter_pre | invariance_counter_pre).sum()}")

    print("\nFine-tuned Model:")
    print(f"  Safe region size (V <= 0): {(V_fine <= 0).mean() * 100:.2f}%")
    print(f"  Constraint counterexamples: {constraint_counter_fine.sum()} points ({constraint_counter_fine.mean() * 100:.3f}%)")
    print(f"  Invariance counterexamples: {invariance_counter_fine.sum()} points ({invariance_counter_fine.mean() * 100:.3f}%)")
    print(f"  Total counterexamples: {(constraint_counter_fine | invariance_counter_fine).sum()}")
    print(f"  TFR (True Feasible Rate): {trf_fine * 100:.2f}%")

    reduction = (constraint_counter_pre | invariance_counter_pre).sum() - (constraint_counter_fine | invariance_counter_fine).sum()
    print("\nImprovement:")
    print(f"  Counterexamples reduced by: {reduction} points")
    print(f"  Reduction rate: {reduction / max((constraint_counter_pre | invariance_counter_pre).sum(), 1) * 100:.2f}%")

    print(f"\nTrue safe region (100-step horizon on slice): {true_safe_region.mean() * 100:.2f}%")
    print("=" * 70)


def main():
    visualize_comparison(
        fine_tune_model_path='../../models/cart_pole/V_finetune_milp.pth',
        name_file='plots/comparison_visualization_milp.png',
        title_suffix='MILP'
    )

    visualize_comparison(
        fine_tune_model_path='../../models/cart_pole/V_finetune_enum_milp.pth',
        name_file='plots/comparison_visualization_enum_milp.png',
        title_suffix='ENUM + MILP'
    )


if __name__ == "__main__":
    main()