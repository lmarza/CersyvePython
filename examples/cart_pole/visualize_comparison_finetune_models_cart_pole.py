"""Visualize CartPole fine-tuned value functions (MILP vs Enum+MILP) on a 2D slice."""

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

N_POINTS = 220
HORIZON = 100
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# Utilities
# ============================================================
def load_state(model, path, device="cpu"):
    state_dict = torch.load(path, map_location=device, weights_only=True)
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


def compute_tfr(V_model, h_model, f_pi_model, n_samples=100_000, horizon=100, device="cpu"):
    """
    Compute the True Feasible Rate (TFR) on the full 4D CartPole state space.

    TFR = (# states with V(x)<=0 AND truly feasible) / (# truly feasible states)

    A state is truly feasible if a 100-step trajectory starting from it
    has no constraint violation (h(x) <= 0 at every step).
    """
    V_model.eval()

    X = uniform(CartPole.x_low, CartPole.x_high, n_samples).astype(np.float32)  # [4, N]
    X_t = torch.from_numpy(X).to(device)  # [4, N]

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

    tfr = n_identified / max(n_truly_feasible, 1)

    return {
        "tfr": tfr,
        "n_truly_feasible": int(n_truly_feasible),
        "n_identified": int(n_identified),
        "truly_feasible_rate": n_truly_feasible / n_samples,
    }


@torch.no_grad()
def eval_value(V_model, pts, n_points, device="cpu"):
    X_torch = torch.from_numpy(pts.astype(np.float32)).to(device)  # [N, 4]
    return V_model(X_torch).cpu().numpy().reshape(n_points, n_points)


@torch.no_grad()
def compute_true_safe_region(pts, h_model, f_pi_model, n_points, horizon=100, device="cpu"):
    """
    Compute true safe region on the chosen 2D slice by forward simulation.
    The slice fixes hidden dimensions to FIXED_STATE.
    """
    print("Computing true safe region (100-step forward reachability on slice)...")
    true_safe_region = np.ones(pts.shape[0], dtype=bool)
    X_current = torch.from_numpy(pts.T.astype(np.float32)).to(device)  # [4, N]

    for step in range(horizon):
        h_current = h_model(X_current.T).cpu().numpy().flatten()
        true_safe_region &= (h_current <= 0)
        X_current = f_pi_model(X_current)

        if step % 20 == 0:
            print(f"  Step {step}/{horizon}: {true_safe_region.mean() * 100:.2f}% states remain safe")

    true_safe_region = true_safe_region.reshape(n_points, n_points)
    print(f"True safe region computed: {true_safe_region.mean() * 100:.2f}% of slice states")
    return true_safe_region


def load_shared_models(device="cpu"):
    """Load CartPole shared models: h, f, pi, closed-loop."""
    print("Loading CartPole shared models...")

    # Data
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

    print("✓ Shared models loaded")
    return x_low, x_high, h_model, f_pi_model


# ============================================================
# Main visualization
# ============================================================
def visualize_comparison():
    np.random.seed(42)
    device = DEVICE

    x_low, x_high, h_model, f_pi_model = load_shared_models(device=device)

    value_hidden_sizes = [64, 64]

    # Evaluate TFR for each seed to select the best one
    seeds = [1, 0, 2]
    trfs_fine_milp = []
    trfs_fine_enum = []

    for seed in seeds:
        # MILP-finetuned model
        V_finetune_milp = create_mlp(CartPole.x_dim, 1, value_hidden_sizes)
        V_finetune_milp_path = f"../../models/cart_pole/V_finetune_milp_seed{seed}.pth"
        print(f"Loading finetuned value function from {V_finetune_milp_path}...")
        V_finetune_milp = load_state(V_finetune_milp, V_finetune_milp_path, device)

        # Enum+MILP-finetuned model
        V_finetune_enum = create_mlp(CartPole.x_dim, 1, value_hidden_sizes)
        V_finetune_enum_path = f"../../models/cart_pole/V_finetune_enum_milp_seed{seed}.pth"
        print(f"Loading finetuned value function from {V_finetune_enum_path}...")
        V_finetune_enum = load_state(V_finetune_enum, V_finetune_enum_path, device)

        # Compute TFR on full 4D space
        trf_fine_milp = compute_tfr(
            V_finetune_milp, h_model, f_pi_model, n_samples=100_000, horizon=100, device=device
        )["tfr"]
        trf_fine_enum = compute_tfr(
            V_finetune_enum, h_model, f_pi_model, n_samples=100_000, horizon=100, device=device
        )["tfr"]

        print(f"  Seed {seed}: MILP TFR={trf_fine_milp * 100:.2f}%, Enum+MILP TFR={trf_fine_enum * 100:.2f}%")

        trfs_fine_milp.append(trf_fine_milp)
        trfs_fine_enum.append(trf_fine_enum)

    # Select best seeds
    best_index_milp = np.argmax(trfs_fine_milp)
    best_index_enum = np.argmax(trfs_fine_enum)

    best_seed_milp = seeds[best_index_milp]
    best_seed_enum = seeds[best_index_enum]

    best_tfr_milp = trfs_fine_milp[best_index_milp]
    best_tfr_enum = trfs_fine_enum[best_index_enum]

    print(f"\nBest MILP seed: {best_seed_milp} (TFR={best_tfr_milp * 100:.2f}%)")
    print(f"Best Enum+MILP seed: {best_seed_enum} (TFR={best_tfr_enum * 100:.2f}%)")

    # Reload best models for visualization
    V_finetune_milp = create_mlp(CartPole.x_dim, 1, value_hidden_sizes)
    V_finetune_milp = load_state(
        V_finetune_milp,
        f"../../models/cart_pole/V_finetune_milp_seed{best_seed_milp}.pth",
        device,
    )

    V_finetune_enum = create_mlp(CartPole.x_dim, 1, value_hidden_sizes)
    V_finetune_enum = load_state(
        V_finetune_enum,
        f"../../models/cart_pole/V_finetune_enum_milp_seed{best_seed_enum}.pth",
        device,
    )

    # Build 2D slice grid
    x1, x2, X1, X2, pts = build_slice_grid(
        x_low=x_low,
        x_high=x_high,
        dims=VIS_DIMS,
        fixed_state=FIXED_STATE,
        n_points=N_POINTS,
    )

    # Compute true safe region on slice
    true_safe_region = compute_true_safe_region(
        pts=pts,
        h_model=h_model,
        f_pi_model=f_pi_model,
        n_points=N_POINTS,
        horizon=HORIZON,
        device=device,
    )

    # Evaluate value functions on slice
    print("\nComputing value functions on CartPole slice...")
    with torch.no_grad():
        V_fine_milp = eval_value(V_finetune_milp, pts, N_POINTS, device)
        V_fine_enum = eval_value(V_finetune_enum, pts, N_POINTS, device)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background heatmap: MILP model
    contour = ax.contourf(X1, X2, V_fine_milp, levels=20, cmap="YlGnBu", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="V(x) [MILP]")

    # True safe boundary
    ax.contour(
        X1, X2, true_safe_region.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=2.5,
        linestyles="-"
    )

    # MILP zero level set
    ax.contour(
        X1, X2, V_fine_milp,
        levels=[0],
        colors="blue",
        linewidths=3,
        linestyles="-"
    )

    # Enum+MILP zero level set
    ax.contour(
        X1, X2, V_fine_enum,
        levels=[0],
        colors="orange",
        linewidths=3,
        linestyles="--"
    )

    ax.set_xlabel("x(1) - Cart Position", fontsize=14)
    ax.set_ylabel("x(3) - Pole Angle", fontsize=14)
    ax.set_title(
        "CartPole: Fine-tuned Value Functions (MILP vs Enum+MILP)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    ax.set_xlim(float(x_low[VIS_DIMS[0]]), float(x_high[VIS_DIMS[0]]))
    ax.set_ylim(float(x_low[VIS_DIMS[1]]), float(x_high[VIS_DIMS[1]]))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=2.5, label="True safe region (100-step slice)"),
        Line2D([0], [0], color="blue", lw=2, label=f"MILP finetuned (TFR={best_tfr_milp * 100:.1f}%)"),
        Line2D([0], [0], color="orange", lw=2, linestyle="--",
               label=f"Enum+MILP finetuned (TFR={best_tfr_enum * 100:.1f}%)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=11,
        framealpha=0.95,
        edgecolor="black"
    )

    plt.tight_layout()

    # Save
    out_path = os.path.join(PLOTS_DIR, "visualize_comparison_finetuned_models.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {out_path}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Value Function Comparison Statistics")
    print("=" * 70)
    print(f"TFR (True Feasible Rate) MILP finetuned: {best_tfr_milp * 100:.2f}% (seed {best_seed_milp})")
    print(f"TFR (True Feasible Rate) Enum+MILP finetuned: {best_tfr_enum * 100:.2f}% (seed {best_seed_enum})")
    print(f"True safe region on displayed slice: {true_safe_region.mean() * 100:.2f}%")
    print(f"Slice dims: {VIS_DIMS}, fixed_state={FIXED_STATE.tolist()}")
    print("=" * 70)


if __name__ == "__main__":
    visualize_comparison()