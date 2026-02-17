"""Visualize pretrained vs finetuned value functions with counterexamples."""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from systems.double_integrator import DoubleIntegrator
from cersyve import create_mlp

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

# Create closed-loop dynamics model
from cersyve.model import create_closed_loop_dynamics_model
import numpy as np
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
V_pretrain = create_mlp(DoubleIntegrator.x_dim, 1, value_hidden_sizes)
V_pretrain_path = '../../models/double_integrator/V_pretrain.pth'
print(f"Loading pretrained value function from {V_pretrain_path}...")
V_pretrain.load_state_dict(torch.load(V_pretrain_path, map_location=device, weights_only=True))
V_pretrain.eval()

# Load finetuned value function
V_finetune = create_mlp(DoubleIntegrator.x_dim, 1, value_hidden_sizes)
V_finetune_path = '../../models/double_integrator/V_finetune.pth'
print(f"Loading finetuned value function from {V_finetune_path}...")
V_finetune.load_state_dict(torch.load(V_finetune_path, map_location=device, weights_only=True))
V_finetune.eval()

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
    V_pre = V_pretrain(X_torch).numpy().reshape(n_points, n_points)  # Pretrained value
    V_fine = V_finetune(X_torch).numpy().reshape(n_points, n_points)  # Finetuned value
    
    # Compute next state values for invariance checking
    X_for_dynamics = torch.from_numpy(X_flat)  # [2, N] for dynamics model
    X_next = f_pi_model(X_for_dynamics).T.cpu().numpy()  # [N, 2]
    X_next_torch = torch.from_numpy(X_next.astype(np.float32))
    V_next_pre = V_pretrain(X_next_torch).numpy().reshape(n_points, n_points)
    V_next_fine = V_finetune(X_next_torch).numpy().reshape(n_points, n_points)

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

# Find counterexamples
# Constraint counterexamples: V <= 0 but H > 0
constraint_counter_pre = (V_pre <= 0) & (H > 0)
constraint_counter_fine = (V_fine <= 0) & (H > 0)

# Invariance counterexamples: V <= 0 but V_next > 0
invariance_counter_pre = (V_pre <= 0) & (V_next_pre > 0)
invariance_counter_fine = (V_fine <= 0) & (V_next_fine > 0)

# Sample counterexamples for visualization (take a subset to avoid clutter)
def sample_counterexamples(mask, n_samples=50):
    indices = np.where(mask.flatten())[0]
    if len(indices) > n_samples:
        indices = np.random.choice(indices, n_samples, replace=False)
    return X_flat[:, indices]

np.random.seed(42)
counter_pre_constraint = sample_counterexamples(constraint_counter_pre, 30)
counter_fine_constraint = sample_counterexamples(constraint_counter_fine, 30)
counter_pre_inv = sample_counterexamples(invariance_counter_pre, 30)
counter_fine_inv = sample_counterexamples(invariance_counter_fine, 30)

# Create single comprehensive plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot finetuned value function as background heatmap
contour = ax.contourf(X1, X2, V_fine, levels=20, cmap='viridis', alpha=0.8)
cbar = plt.colorbar(contour, ax=ax, label='Fine-tuned Value V(x)')

# Plot true safe region boundary (black line - 100-step forward reachability)
ax.contour(X1, X2, true_safe_region.astype(float), levels=[0.5], colors='black', linewidths=2.5, linestyles='-')

# Plot pre-trained safe region V=0 (orange line)
ax.contour(X1, X2, V_pre, levels=[0], colors='orange', linewidths=2, linestyles='-')

# Plot verified safe region V=0 (blue line)
ax.contour(X1, X2, V_fine, levels=[0], colors='blue', linewidths=2, linestyles='-')

# Plot counterexamples from pre-training (lighter red dots)
if counter_pre_constraint.shape[1] > 0 or counter_pre_inv.shape[1] > 0:
    all_pre_counter = np.hstack([counter_pre_constraint, counter_pre_inv]) if counter_pre_constraint.shape[1] > 0 and counter_pre_inv.shape[1] > 0 else (counter_pre_constraint if counter_pre_constraint.shape[1] > 0 else counter_pre_inv)
    ax.scatter(all_pre_counter[0, :], all_pre_counter[1, :], 
               c='red', s=60, alpha=0.6, edgecolors='darkred', linewidth=1.5, 
               marker='o', zorder=5)

# Compute and plot next states for counterexamples (darker red dots)
if counter_pre_constraint.shape[1] > 0 or counter_pre_inv.shape[1] > 0:
    all_pre_counter = np.hstack([counter_pre_constraint, counter_pre_inv]) if counter_pre_constraint.shape[1] > 0 and counter_pre_inv.shape[1] > 0 else (counter_pre_constraint if counter_pre_constraint.shape[1] > 0 else counter_pre_inv)
    with torch.no_grad():
        counter_torch = torch.from_numpy(all_pre_counter.astype(np.float32))
        counter_next = f_pi_model(counter_torch).cpu().numpy()  # [2, N]
    ax.scatter(counter_next[0, :], counter_next[1, :], 
               c='darkred', s=60, alpha=0.8, edgecolors='black', linewidth=1.5, 
               marker='o', zorder=5)
    
    # Draw arrows from counterexamples to their next states
    for i in range(min(all_pre_counter.shape[1], 30)):  # Limit arrows to avoid clutter
        ax.arrow(all_pre_counter[0, i], all_pre_counter[1, i],
                counter_next[0, i] - all_pre_counter[0, i],
                counter_next[1, i] - all_pre_counter[1, i],
                head_width=0.02, head_length=0.03, fc='red', ec='darkred',
                alpha=0.5, linewidth=1, zorder=4)

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
    Line2D([0], [0], color='orange', lw=2, label='Pre-trained region'),
    Line2D([0], [0], color='blue', lw=2, label='Fine-Tuned region'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
           markersize=9, label='Counterexample', markeredgecolor='darkred', markeredgewidth=1.5),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
           markersize=9, label='Next state', markeredgecolor='black', markeredgewidth=1.5)
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
          framealpha=0.95, edgecolor='black')

plt.tight_layout()

# Save figure
save_path = 'comparison_visualization.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {save_path}")

# Print statistics
print("\n" + "="*70)
print("Value Function Comparison Statistics")
print("="*70)

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

print("\nImprovement:")
reduction = (constraint_counter_pre | invariance_counter_pre).sum() - (constraint_counter_fine | invariance_counter_fine).sum()
print(f"  Counterexamples reduced by: {reduction} points")
print(f"  Reduction rate: {reduction / max((constraint_counter_pre | invariance_counter_pre).sum(), 1) * 100:.2f}%")

# True safe region (100-step forward reachability)
print(f"\nTrue safe region (100-step horizon): {true_safe_region.mean() * 100:.2f}%")

print("="*70)
