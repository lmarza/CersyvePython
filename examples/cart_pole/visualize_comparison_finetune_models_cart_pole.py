"""Visualize CartPole finetuned value functions together on the same slice."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from systems.cart_pole import CartPole
from cersyve import create_mlp
from cersyve.utils import uniform
from cersyve.model import create_closed_loop_dynamics_model


VIS_DIMS = (0, 2)  # position, angle
FIXED_STATE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
N_POINTS = 220
HORIZON = 100


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
    identified = (V <= 0) & truly_feasible
    n_identified = identified.sum()

    return n_identified / max(n_truly_feasible, 1)


def build_slice_grid(x_low, x_high, dims=(0, 2), fixed_state=None, n_points=200):
    if fixed_state is None:
        fixed_state = np.zeros(CartPole.x_dim, dtype=np.float32)

    x_axis = np.linspace(x_low[dims[0]], x_high[dims[0]], n_points, dtype=np.float32)
    y_axis = np.linspace(x_low[dims[1]], x_high[dims[1]], n_points, dtype=np.float32)
    X1, X2 = np.meshgrid(x_axis, y_axis)

    pts = np.repeat(fixed_state[None, :], n_points * n_points, axis=0).astype(np.float32)
    pts[:, dims[0]] = X1.reshape(-1)
    pts[:, dims[1]] = X2.reshape(-1)

    return X1, X2, pts


def load_models(device="cpu"):
    os.makedirs("plots", exist_ok=True)

    data_path = "../../data/cart_pole_data.npz"
    print(f"Loading data from {data_path} ...")
    data_npz = np.load(data_path)
    data = {}
    for k in data_npz.keys():
        arr = data_npz[k]
        if np.issubdtype(arr.dtype, np.floating):
            data[k] = arr.astype(np.float32)
        else:
            data[k] = arr

    x_low = np.asarray(CartPole.x_low, dtype=np.float32)
    x_high = np.asarray(CartPole.x_high, dtype=np.float32)

    print(f"Using device: {device}")

    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_model.load_state_dict(torch.load("../../models/cart_pole/h.pth", map_location=device))
    h_model.to(device).float().eval()

    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_model.load_state_dict(torch.load("../../models/cart_pole/f.pth", map_location=device))
    f_model.to(device).float().eval()

    pi_model = CartPole.create_policy().to(device).float().eval()
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, x_low, x_high, CartPole.u_dim
    ).to(device).float().eval()

    V_milp = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_milp.load_state_dict(torch.load("../../models/cart_pole/V_finetune_milp.pth", map_location=device))
    V_milp.to(device).float().eval()

    V_enum = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_enum.load_state_dict(torch.load("../../models/cart_pole/V_finetune_enum_milp.pth", map_location=device))
    V_enum.to(device).float().eval()

    return x_low, x_high, h_model, f_pi_model, V_milp, V_enum


def main():
    device = "cpu"
    x_low, x_high, h_model, f_pi_model, V_milp, V_enum = load_models(device=device)

    X1, X2, pts = build_slice_grid(
        x_low=x_low,
        x_high=x_high,
        dims=VIS_DIMS,
        fixed_state=FIXED_STATE,
        n_points=N_POINTS,
    )

    with torch.no_grad():
        X_torch = torch.from_numpy(pts.astype(np.float32)).to(device)
        V_milp_vals = V_milp(X_torch).cpu().numpy().reshape(N_POINTS, N_POINTS)
        V_enum_vals = V_enum(X_torch).cpu().numpy().reshape(N_POINTS, N_POINTS)

    # true safe region on slice
    true_safe_region = np.ones(pts.shape[0], dtype=bool)
    with torch.no_grad():
        X_current = torch.from_numpy(pts.T.astype(np.float32)).to(device)
        for step in range(HORIZON):
            h_current = h_model(X_current.T).cpu().numpy().flatten()
            true_safe_region &= (h_current <= 0)
            X_current = f_pi_model(X_current)
            if step % 20 == 0:
                print(f"  Step {step}/{HORIZON}: {true_safe_region.mean()*100:.2f}% states remain safe")

    true_safe_region = true_safe_region.reshape(N_POINTS, N_POINTS)

    trf_milp = compute_tfr(V_milp, h_model, f_pi_model, n_samples=100_000, horizon=100, device=device)
    trf_enum = compute_tfr(V_enum, h_model, f_pi_model, n_samples=100_000, horizon=100, device=device)

    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(X1, X2, V_milp_vals, levels=20, cmap="YlGnBu", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="MILP Fine-tuned Value V(x)")

    ax.contour(X1, X2, true_safe_region.astype(float), levels=[0.5], colors="black", linewidths=2.5)
    ax.contour(X1, X2, V_milp_vals, levels=[0], colors="blue", linewidths=3)
    ax.contour(X1, X2, V_enum_vals, levels=[0], colors="orange", linewidths=3, linestyles="--")

    ax.set_xlabel("x(1) - Cart Position", fontsize=14)
    ax.set_ylabel("x(3) - Pole Angle", fontsize=14)
    ax.set_title(
        "CartPole: Fine-tuned Value Functions Comparison",
        fontsize=15, fontweight="bold", pad=20
    )
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color="black", lw=2.5, label="True region"),
        Line2D([0], [0], color="blue", lw=2.5, label="Fine-Tuned MILP region"),
        Line2D([0], [0], color="orange", lw=2.5, linestyle="--", label="Fine-Tuned ENUM+MILP region"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.95, edgecolor="black")

    plt.tight_layout()
    plt.savefig("plots/visualize_comparison_finetuned_models.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved to: plots/visualize_comparison_finetuned_models.png")

    print("\n" + "=" * 70)
    print("Value Function Comparison Statistics")
    print("=" * 70)
    print(f"TRF (True Feasible Rate) MILP finetuned: {trf_milp * 100:.2f}%")
    print(f"TRF (True Feasible Rate) ENUM+MILP finetuned: {trf_enum * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()