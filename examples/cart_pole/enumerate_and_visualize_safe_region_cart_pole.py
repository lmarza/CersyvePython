"""
Enumerate unsafe regions (4D) and visualize on the (y, theta) slice with ydot=0, thdot=0.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from systems.cart_pole import CartPole
from cersyve import create_mlp, create_closed_loop_dynamics_model, BinaryDecisionWrapperCBF, enumerate_unsafe_regions
from cersyve.enumeration import is_inside

def main():
    models_dir = "../../models/cart_pole/"
    logs_dir = "../../logs/cart_pole/"
    data_path = "../../data/cart_pole_data.npz"
    os.makedirs("plots", exist_ok=True)

    device = "cpu"
    data_npz = np.load(data_path)
    data = {k: data_npz[k] for k in data_npz.keys()}

    # load models
    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_model.load_state_dict(torch.load(os.path.join(models_dir, "f.pth"), map_location=device))
    f_model.eval()

    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_model.load_state_dict(torch.load(os.path.join(models_dir, "h.pth"), map_location=device))
    h_model.eval()

    V_model = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_model.load_state_dict(torch.load(os.path.join(models_dir, "V_pretrain.pth"), map_location=device))
    V_model.eval()

    pi_model = CartPole.create_policy().eval()
    f_pi_model = create_closed_loop_dynamics_model(f_model, pi_model, data, CartPole.x_low, CartPole.x_high, CartPole.u_dim)
    f_pi_model.eval()

    # enumeration
    wrapper = BinaryDecisionWrapperCBF(V_model, h_model, f_pi_model, tol=0.0)
    input_domain = np.stack([CartPole.x_low, CartPole.x_high], axis=1)  # (4,2)

    unsafe_regions, stats = enumerate_unsafe_regions(
        wrapper_model=wrapper,
        input_domain=input_domain,
        n_training_samples=60_000,
        n_decision_trees=800,
        max_depth=14,
        enumerate_safe_regions=False,
        incremental_coverage=True,
        wilks_filtering=True,
        wilks_samples=10,
        sample_size_evaluation=60_000,
        desired_coverage=70.0,
        digits_to_consider=0,
        seed=42,
        verbose=True,
        device=device
    )
    print("Unsafe regions:", len(unsafe_regions), "coverage:", stats["coverage"], "error:", stats["error_rate"])

    # visualize slice y,th with ydot=0, thdot=0
    n = 220
    y = np.linspace(CartPole.x_low[0], CartPole.x_high[0], n)
    th = np.linspace(CartPole.x_low[1], CartPole.x_high[1], n)
    Y, TH = np.meshgrid(y, th)

    X = np.zeros((4, n*n), dtype=np.float32)
    X[0, :] = Y.flatten()
    X[1, :] = TH.flatten()
    X[2, :] = 0.0
    X[3, :] = 0.0

    with torch.no_grad():
        V_grid = V_model(torch.from_numpy(X.T)).cpu().numpy().reshape(n, n)

    # true safe via rollout on slice
    horizon = 100
    Xcur = torch.from_numpy(X)
    true_safe = torch.ones(Xcur.shape[1], dtype=torch.bool)
    with torch.no_grad():
        for _ in range(horizon):
            hcur = h_model(Xcur.T).squeeze(1)
            true_safe &= (hcur <= 0)
            Xcur = f_pi_model(Xcur)
    true_safe_map = true_safe.numpy().reshape(n, n).astype(float)

    # inside unsafe rectangles on the slice:
    # keep only regions that include ydot=0 and thdot=0 in their bounds (dims 2,3)
    slice_regions = [r for r in unsafe_regions if (r[2,0] <= 0 <= r[2,1] and r[3,0] <= 0 <= r[3,1])]
    inside = is_inside(X[:2, :].T, [r[:2, :] for r in slice_regions]) if slice_regions else np.zeros(n*n, dtype=bool)
    inside_map = inside.reshape(n, n).astype(float)

    plt.figure(figsize=(10,8))
    plt.contourf(Y, TH, V_grid, levels=25, alpha=0.85)
    plt.colorbar(label="V(x) on slice (ydot=0, thdot=0)")
    plt.contour(Y, TH, true_safe_map, levels=[0.5], colors="black", linewidths=2.5)
    plt.contour(Y, TH, V_grid, levels=[0], colors="blue", linewidths=2.5)
    plt.contour(Y, TH, inside_map, levels=[0.5], colors="red", linewidths=2.0, linestyles="--")

    plt.xlabel("y (cart position)")
    plt.ylabel("theta (pole angle)")
    plt.title("CartPole slice: true safe (black), V=0 (blue), enumerated unsafe-projection (red dashed)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = "plots/cart_pole_enum_slice.png"
    plt.savefig(out, dpi=300)
    print("Saved:", out)

if __name__ == "__main__":
    main()