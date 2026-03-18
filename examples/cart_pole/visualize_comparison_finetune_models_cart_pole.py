"""CartPole: compare MILP and ENUM+MILP fine-tuned value functions on the same slice."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from systems.cart_pole import CartPole
from cersyve import create_mlp
from cersyve.utils import uniform
from cersyve.model import create_closed_loop_dynamics_model


DEVICE = "cpu"
VIS_DIMS = (0, 2)  # cart position, pole angle
FIXED_STATE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

N_POINTS = 500
HORIZON = 100
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_state(model, path, device="cpu"):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).float().eval()
    return model


def compute_tfr(V_model, h_model, f_pi_model, n_samples=100_000, horizon=100, device="cpu"):
    V_model.eval()
    X = uniform(CartPole.x_low, CartPole.x_high, n_samples).astype(np.float32)
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        V = V_model(X_t.T).T.cpu().numpy()[0, :]
        truly_feasible = np.ones(n_samples, dtype=bool)
        X_current = X_t.clone()

        for _ in range(horizon):
            h_current = h_model(X_current.T).T.cpu().numpy()[0, :]
            truly_feasible &= (h_current <= 0)
            X_current = f_pi_model(X_current)

    n_truly_feasible = truly_feasible.sum()
    n_identified = ((V <= 0) & truly_feasible).sum()
    return n_identified / max(n_truly_feasible, 1)


def build_slice_grid(x_low, x_high, dims=(0, 2), fixed_state=None, n_points=400):
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
    safe = np.ones(pts.shape[0], dtype=bool)
    X_current = torch.from_numpy(pts.T.astype(np.float32)).to(device)

    for step in range(horizon):
        h_val = h_model(X_current.T).cpu().numpy().flatten()
        safe &= (h_val <= 0.0)
        X_current = f_pi_model(X_current)

        if step % 20 == 0:
            print(f"  Step {step}/{horizon}: {safe.mean()*100:.2f}% points remain safe")

    return safe.reshape(n_points, n_points)


def load_models(device="cpu"):
    print("Loading CartPole data and models...")

    data_npz = np.load("../../data/cart_pole_data.npz")
    data = {}
    for k in data_npz.keys():
        arr = data_npz[k]
        data[k] = arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr

    x_low = np.asarray(CartPole.x_low, dtype=np.float32)
    x_high = np.asarray(CartPole.x_high, dtype=np.float32)

    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_model = load_state(h_model, "../../models/cart_pole/h.pth", device)

    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_model = load_state(f_model, "../../models/cart_pole/f.pth", device)

    pi_model = CartPole.create_policy().to(device).float().eval()

    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, x_low, x_high, CartPole.u_dim
    ).to(device).float().eval()

    V_milp = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_milp = load_state(V_milp, "../../models/cart_pole/V_finetune_milp.pth", device)

    V_enum = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_enum = load_state(V_enum, "../../models/cart_pole/V_finetune_enum_milp.pth", device)

    print("✓ Models loaded\n")
    return x_low, x_high, h_model, f_pi_model, V_milp, V_enum


def main():
    x_low, x_high, h_model, f_pi_model, V_milp, V_enum = load_models(device=DEVICE)

    xs, ys, X1, X2, pts = build_slice_grid(
        x_low=x_low,
        x_high=x_high,
        dims=VIS_DIMS,
        fixed_state=FIXED_STATE,
        n_points=N_POINTS,
    )

    print("Computing values on slice...")
    V_milp_vals = eval_value(V_milp, pts, N_POINTS, DEVICE)
    V_enum_vals = eval_value(V_enum, pts, N_POINTS, DEVICE)

    print("Computing true safe region...")
    true_safe_region = compute_true_safe_region(
        pts=pts,
        h_model=h_model,
        f_pi_model=f_pi_model,
        n_points=N_POINTS,
        horizon=HORIZON,
        device=DEVICE,
    )

    tfr_milp = compute_tfr(V_milp, h_model, f_pi_model, n_samples=100_000, horizon=100, device=DEVICE)
    tfr_enum = compute_tfr(V_enum, h_model, f_pi_model, n_samples=100_000, horizon=100, device=DEVICE)

    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(
        X1, X2, V_milp_vals,
        levels=30,
        cmap="YlGnBu",
        alpha=0.86
    )
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Fine-tuned Value V(x)", fontsize=13)

    # true region
    ax.contour(
        X1, X2, true_safe_region.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=2.4
    )

    # MILP fine-tuned
    ax.contour(
        X1, X2, V_milp_vals,
        levels=[0.0],
        colors="blue",
        linewidths=2.8
    )

    # ENUM fine-tuned
    ax.contour(
        X1, X2, V_enum_vals,
        levels=[0.0],
        colors="orange",
        linewidths=2.8,
        linestyles="--"
    )

    ax.set_xlabel("x(1) - Position", fontsize=14)
    ax.set_ylabel("x(3) - Pole Angle", fontsize=14)
    ax.set_title(
        "CartPole: Pre-trained vs Fine-tuned Value Functions",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.grid(True, alpha=0.25)
    ax.set_xlim(float(x_low[0]), float(x_high[0]))
    ax.set_ylim(float(x_low[2]), float(x_high[2]))

    legend_elements = [
        Line2D([0], [0], color="black", lw=2.4, label="True region"),
        Line2D([0], [0], color="blue", lw=2.8, label="Fine-Tuned MILP region"),
        Line2D([0], [0], color="orange", lw=2.8, linestyle="--", label="Fine-Tuned ENUM region"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.95, edgecolor="black")

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "visualize_comparison_finetuned_models.png")
    plt.savefig(out_file, dpi=320, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_file}")
    print("=" * 70)
    print("Fine-tuned comparison statistics")
    print("=" * 70)
    print(f"TFR MILP: {tfr_milp * 100:.2f}%")
    print(f"TFR ENUM: {tfr_enum * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()