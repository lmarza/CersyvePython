"""Visualize pretrained vs finetuned value functions for the unicycle system."""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import matplotlib.pyplot as plt
from systems.unicycle import Unicycle
from cersyve import create_mlp
from cersyve.utils import uniform
from cersyve.model import create_closed_loop_dynamics_model
import numpy as np


def compute_tfr(V_model, h_model, f_pi_model, n_samples=100_000, horizon=100, device='cpu',
                return_missed=False):
    """Compute the True Feasible Rate (TFR).

    TFR = (# states with V(x)<=0 AND truly feasible) / (# truly feasible states)

    A state is truly feasible if a 100-step trajectory starting from it
    has no constraint violation (h(x) <= 0 at every step).

    If return_missed=True, also returns the truly feasible states with V(x)>0.
    """
    V_model.eval()
    X = uniform(Unicycle.x_low, Unicycle.x_high, n_samples)
    X_t = torch.from_numpy(X).to(device)  # [x_dim, N]

    with torch.no_grad():
        V = V_model(X_t.T).T.cpu().numpy()[0, :]  # [N]

        # Forward simulate and check constraints along the trajectory
        truly_feasible = np.ones(n_samples, dtype=bool)
        X_current = X_t.clone()  # [x_dim, N]
        for step in range(horizon):
            h_current = h_model(X_current.T).T.cpu().numpy()[0, :]
            truly_feasible &= (h_current <= 0)
            X_current = f_pi_model(X_current)

    n_truly_feasible = truly_feasible.sum()
    identified = (V <= 0) & truly_feasible
    n_identified = identified.sum()
    missed = truly_feasible & (V > 0)  # truly feasible but V says unsafe

    tfr = n_identified / max(n_truly_feasible, 1)
    result = {
        'tfr': tfr,
        'n_truly_feasible': n_truly_feasible,
        'n_identified': n_identified,
        'n_missed': missed.sum(),
        'truly_feasible_rate': n_truly_feasible / n_samples,
    }
    if return_missed:
        result['X_missed'] = X[:, missed]  # [x_dim, n_missed]
        result['V_missed'] = V[missed]     # V values at missed states
    return result


def visualize_comparison():

    np.random.seed(42)
    # Device
    device = 'cpu'

    # Load constraint model
    constraint_hidden_sizes = [16]
    h_model = create_mlp(Unicycle.x_dim, 1, constraint_hidden_sizes)
    h_model_path = '../../models/unicycle/h.pth'
    print(f"Loading constraint model from {h_model_path}...")
    h_model.load_state_dict(torch.load(h_model_path, map_location=device, weights_only=True))
    h_model.eval()

    # Load dynamics model
    dynamics_hidden_sizes = [32, 32]
    f_model = create_mlp(Unicycle.x_dim + Unicycle.u_dim, Unicycle.x_dim, dynamics_hidden_sizes)
    f_model_path = '../../models/unicycle/f.pth'
    print(f"Loading dynamics model from {f_model_path}...")
    f_model.load_state_dict(torch.load(f_model_path, map_location=device, weights_only=True))
    f_model.eval()

    data_path = '../../data/unicycle_data.npz'
    data_npz = np.load(data_path)
    data = {key: data_npz[key] for key in data_npz.keys()}

    pi_model = Unicycle.create_policy()
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, Unicycle.x_low, Unicycle.x_high, Unicycle.u_dim
    )
    f_pi_model.eval()

    # Fixed velocity slices for visualization (v dimension)
    v_slices = [-0.5, 0.0, 0.5]

    # Create grid for visualization over (xo, yo)
    n_points = 150
    xo_vals = np.linspace(Unicycle.x_low[1], Unicycle.x_high[1], n_points)
    yo_vals = np.linspace(Unicycle.x_low[2], Unicycle.x_high[2], n_points)
    Xo, Yo = np.meshgrid(xo_vals, yo_vals)

    # Evaluate TFR for each seed to select the best one
    #seeds = [42, 2101, 70]
    seeds = [1, 0, 2]
    trfs_fine_milp = []
    trfs_fine_enum = []
    value_hidden_sizes = [32, 32]

    for seed in seeds:
        # Load MILP-finetuned value function
        V_finetune_milp = create_mlp(Unicycle.x_dim, 1, value_hidden_sizes)
        V_finetune_milp_path = f'../../models/unicycle/V_finetune_milp_{seed}.pth'
        print(f"Loading finetuned value function from {V_finetune_milp_path}...")
        V_finetune_milp.load_state_dict(torch.load(V_finetune_milp_path, map_location=device, weights_only=True))
        V_finetune_milp.eval()

        # Load enum+MILP-finetuned value function
        V_finetune_enum = create_mlp(Unicycle.x_dim, 1, value_hidden_sizes)
        V_finetune_enum_path = f'../../models/unicycle/V_finetune_enum_milp_{seed}.pth'
        print(f"Loading finetuned value function from {V_finetune_enum_path}...")
        V_finetune_enum.load_state_dict(torch.load(V_finetune_enum_path, map_location=device, weights_only=True))
        V_finetune_enum.eval()

        # Compute TFR for finetuned models
        trf_fine_milp = compute_tfr(V_finetune_milp, h_model, f_pi_model, n_samples=100_000, horizon=100)['tfr']
        trf_fine_enum = compute_tfr(V_finetune_enum, h_model, f_pi_model, n_samples=100_000, horizon=100)['tfr']
        print(f"  Seed {seed}: MILP TFR={trf_fine_milp*100:.2f}%, Enum+MILP TFR={trf_fine_enum*100:.2f}%")

        trfs_fine_milp.append(trf_fine_milp)
        trfs_fine_enum.append(trf_fine_enum)

    # Select the best seed based on TFR for visualization
    best_index_milp = np.argmax(trfs_fine_milp)
    best_index_enum = np.argmax(trfs_fine_enum)
    best_tfr_milp = trfs_fine_milp[best_index_milp]
    best_tfr_enum = trfs_fine_enum[best_index_enum]
    print(f"\nBest MILP seed: {seeds[best_index_milp]} (TFR={best_tfr_milp*100:.2f}%)")
    print(f"Best Enum+MILP seed: {seeds[best_index_enum]} (TFR={best_tfr_enum*100:.2f}%)")

    # Reload the best models for visualization
    V_finetune_milp = create_mlp(Unicycle.x_dim, 1, value_hidden_sizes)
    V_finetune_milp_path = f'../../models/unicycle/V_finetune_milp_{seeds[best_index_milp]}.pth'
    V_finetune_milp.load_state_dict(torch.load(V_finetune_milp_path, map_location=device, weights_only=True))
    V_finetune_milp.eval()

    V_finetune_enum = create_mlp(Unicycle.x_dim, 1, value_hidden_sizes)
    V_finetune_enum_path = f'../../models/unicycle/V_finetune_enum_milp_{seeds[best_index_enum]}.pth'
    V_finetune_enum.load_state_dict(torch.load(V_finetune_enum_path, map_location=device, weights_only=True))
    V_finetune_enum.eval()

    # Create figure: one column per v-slice, rows for MILP and Enum+MILP
    fig, axes = plt.subplots(1, len(v_slices), figsize=(5 * len(v_slices), 5))
    fig.suptitle('Unicycle: Fine-tuned Value Functions (MILP vs Enum+MILP)',
                 fontsize=15, fontweight='bold')

    for col, v_fixed in enumerate(v_slices):
        ax = axes[col]

        # Build flat state array [3, N] with fixed v
        X_flat = np.stack([
            np.full(n_points * n_points, v_fixed, dtype=np.float32),
            Xo.flatten().astype(np.float32),
            Yo.flatten().astype(np.float32),
        ], axis=0)  # [3, N]

        # Compute true safe region by forward simulation
        horizon = 100
        true_safe_region = np.ones(n_points * n_points, dtype=bool)
        with torch.no_grad():
            X_current = torch.from_numpy(X_flat)  # [3, N]
            for step in range(horizon):
                h_current = h_model(X_current.T).T.cpu().numpy()[0, :]
                true_safe_region &= (h_current <= 0)
                X_current = f_pi_model(X_current)
        true_safe_region = true_safe_region.reshape(n_points, n_points)

        # Evaluate value functions on grid
        with torch.no_grad():
            X_torch = torch.from_numpy(X_flat.T)  # [N, 3]
            V_fine_milp_grid = V_finetune_milp(X_torch).numpy().reshape(n_points, n_points)
            V_fine_enum_grid = V_finetune_enum(X_torch).numpy().reshape(n_points, n_points)

        # Plot MILP value function as background heatmap
        contour = ax.contourf(Xo, Yo, V_fine_milp_grid, levels=20, cmap='YlGnBu', alpha=0.8)
        plt.colorbar(contour, ax=ax, label='V(x) [MILP]')

        # True safe region boundary
        ax.contour(Xo, Yo, true_safe_region.astype(float), levels=[0.5],
                   colors='black', linewidths=2.5, linestyles='-')

        # MILP value function zero level set
        ax.contour(Xo, Yo, V_fine_milp_grid, levels=[0],
                   colors='blue', linewidths=3, linestyles='-')

        # Enum+MILP value function zero level set
        ax.contour(Xo, Yo, V_fine_enum_grid, levels=[0],
                   colors='orange', linewidths=3, linestyles='--')

        ax.set_xlabel(r'$x_o$ - Obstacle x', fontsize=12)
        ax.set_ylabel(r'$y_o$ - Obstacle y', fontsize=12)
        ax.set_title(f'v = {v_fixed:.1f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2.5, label='True safe region'),
        Line2D([0], [0], color='blue', lw=2, label=f'MILP finetuned (TFR={best_tfr_milp*100:.1f}%)'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--',
               label=f'Enum+MILP finetuned (TFR={best_tfr_enum*100:.1f}%)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
               framealpha=0.95, edgecolor='black', bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Save figure
    plt.savefig("plots/visualize_comparison_finetuned_models.png", dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: plots/visualize_comparison_finetuned_models.png")

    # Print statistics
    print("\n" + "=" * 70)
    print("Value Function Comparison Statistics")
    print("=" * 70)
    print(f"TFR (True Feasible Rate) MILP finetuned: {best_tfr_milp*100:.2f}% (seed {seeds[best_index_milp]})")
    print(f"TFR (True Feasible Rate) Enum+MILP finetuned: {best_tfr_enum*100:.2f}% (seed {seeds[best_index_enum]})")
    print("=" * 70)


if __name__ == "__main__":
    visualize_comparison()
