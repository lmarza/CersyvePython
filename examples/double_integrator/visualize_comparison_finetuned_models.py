"""Visualize pretrained vs finetuned value functions with counterexamples."""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import matplotlib.pyplot as plt
from systems.double_integrator import DoubleIntegrator
from cersyve import create_mlp
from cersyve.utils import uniform
# Create closed-loop dynamics model
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
    X = uniform(DoubleIntegrator.x_low, DoubleIntegrator.x_high, n_samples)
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
    h_model = create_mlp(DoubleIntegrator.x_dim, 1, constraint_hidden_sizes)
    h_model_path = '../../models/double_integrator/h.pth'
    print(f"Loading constraint model from {h_model_path}...")
    h_model.load_state_dict(torch.load(h_model_path, map_location=device, weights_only=True))
    h_model.eval()

    # Load dynamics model
    dynamics_hidden_sizes = [32, 32]
    f_model = create_mlp(DoubleIntegrator.x_dim + DoubleIntegrator.u_dim, DoubleIntegrator.x_dim, dynamics_hidden_sizes)
    f_model_path = '../../models/double_integrator/f.pth'
    print(f"Loading dynamics model from {f_model_path}...")
    f_model.load_state_dict(torch.load(f_model_path, map_location=device, weights_only=True))
    f_model.eval()


    data_path = '../../data/double_integrator_data.npz'
    data_npz = np.load(data_path)
    data = {key: data_npz[key] for key in data_npz.keys()}

    pi_model = DoubleIntegrator.create_policy()
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, DoubleIntegrator.x_low, DoubleIntegrator.x_high, DoubleIntegrator.u_dim
    )
    f_pi_model.eval()

    # Load pretrained value function
    value_hidden_sizes = [32, 32]
    V_finetune_milp = create_mlp(DoubleIntegrator.x_dim, 1, value_hidden_sizes)
    V_finetune_milp_path = '../../models/double_integrator/V_finetune_milp.pth'
    print(f"Loading finetuned value function from {V_finetune_milp_path}...")
    V_finetune_milp.load_state_dict(torch.load(V_finetune_milp_path, map_location=device, weights_only=True))
    V_finetune_milp.eval()
   

    # Load finetuned value function
    V_finetune_enum = create_mlp(DoubleIntegrator.x_dim, 1, value_hidden_sizes)
    V_finetune_enum_path = '../../models/double_integrator/V_finetune_enum_milp.pth'
    print(f"Loading finetuned value function from {V_finetune_enum_path}...")
    V_finetune_enum.load_state_dict(torch.load(V_finetune_enum_path, map_location=device, weights_only=True))
    V_finetune_enum.eval()

    # Create grid for visualization
    n_points = 200
    x1 = np.linspace(DoubleIntegrator.x_low[0], DoubleIntegrator.x_high[0], n_points)
    x2 = np.linspace(DoubleIntegrator.x_low[1], DoubleIntegrator.x_high[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)

    # Flatten grid for batch evaluation
    X_flat = np.stack([X1.flatten(), X2.flatten()], axis=0).astype(np.float32)  # [2, N]

    # Compute values
    print("Computing value functions and constraints...")
    with torch.no_grad():
        X_torch = torch.from_numpy(X_flat.T)  # [N, 2]
        H = h_model(X_torch).numpy().reshape(n_points, n_points)  # True constraint
        V_fine_milp = V_finetune_milp(X_torch).numpy().reshape(n_points, n_points)  # Finetuned value
        V_fine_enum = V_finetune_enum(X_torch).numpy().reshape(n_points, n_points)  # Finetuned value

        
        # Compute next state values for invariance checking
        X_for_dynamics = torch.from_numpy(X_flat)  # [2, N] for dynamics model
        X_next = f_pi_model(X_for_dynamics).T.cpu().numpy()  # [N, 2]
        X_next_torch = torch.from_numpy(X_next.astype(np.float32))
        V_next_fine_milp = V_finetune_milp(X_next_torch).numpy().reshape(n_points, n_points)
        V_next_fine_enum = V_finetune_enum(X_next_torch).numpy().reshape(n_points, n_points)

    # Compute true safe region by forward simulation
    print("Computing true safe region (100-step forward reachability)...")
    horizon = 100
    true_safe_region = np.ones(n_points * n_points, dtype=bool)

    with torch.no_grad():
        X_current = torch.from_numpy(X_flat)  # [2, N]
        
        for step in range(horizon):
            # Check constraint at current state
            X_current_for_h = torch.from_numpy(X_current.numpy().T)  # [N, 2]
            h_current = h_model(X_current_for_h).numpy().flatten()
            
            # Mark states that violate constraint
            true_safe_region &= (h_current <= 0)
            
            # Simulate one step forward
            X_current = f_pi_model(X_current)
            
            if step % 20 == 0:
                print(f"  Step {step}/{horizon}: {true_safe_region.mean()*100:.2f}% states remain safe")

    true_safe_region = true_safe_region.reshape(n_points, n_points)
    print(f"True safe region computed: {true_safe_region.mean()*100:.2f}% of states")
    print()

    # compute TRF for fine-tuned models
    trf_fine_milp = compute_tfr(V_finetune_milp, h_model, f_pi_model, n_samples=100_000, horizon=100)['tfr']
    trf_fine_enum = compute_tfr(V_finetune_enum, h_model, f_pi_model, n_samples=100_000, horizon=100)['tfr']
    # Create single comprehensive plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot finetuned value function as background heatmap
    contour = ax.contourf(X1, X2, V_fine_milp, levels=20, cmap='YlGnBu', alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax, label='Fine-tuned Value V(x)')

    # Plot true safe region boundary (black line - 100-step forward reachability)
    ax.contour(X1, X2, true_safe_region.astype(float), levels=[0.5], colors='black', linewidths=2.5, linestyles='-')

    # Plot verified safe region V=0 (blue line)
    ax.contour(X1, X2, V_fine_milp, levels=[0], colors='blue', linewidths=3, linestyles='-')

     # Plot verified safe region V=0 (blue line)
    ax.contour(X1, X2, V_fine_enum, levels=[0], colors='orange', linewidths=3, linestyles='--')

    ax.set_xlabel('$x$(1) - Position', fontsize=14)
    ax.set_ylabel('$x$(2) - Velocity', fontsize=14)
    ax.set_title('Double Integrator: Pre-trained vs Fine-tuned Value Functions', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Create comprehensive legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    legend_elements = [
        Line2D([0], [0], color='black', lw=2.5, label='True region'),
        Line2D([0], [0], color='blue', lw=2, label='Fine-Tuned MILP region'),
        Line2D([0], [0], color='orange', lw=2, label='Fine-Tuned ENUM region'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
            framealpha=0.95, edgecolor='black')

    plt.tight_layout()

    # Save figure
    plt.savefig("plots/visualize_comparison_finetuned_models.png", dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: plots/visualize_comparison_finetuned_models.png")

    # Print statistics
    print("\n" + "="*70)
    print("Value Function Comparison Statistics")
    print("="*70)

    print("TRF (True Feasible Rate) MILP finetuned:", f"{trf_fine_milp*100:.2f}%")
    print("TRF (True Feasible Rate) ENUM finetuned:", f"{trf_fine_enum*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    visualize_comparison()