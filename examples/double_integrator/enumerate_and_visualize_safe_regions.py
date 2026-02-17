"""Enumerate probable unsafe regions and visualize them with the pre-trained feasible region."""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from systems.double_integrator import DoubleIntegrator
from cersyve import (
    create_mlp, 
    create_closed_loop_dynamics_model, 
    BinaryDecisionWrapperCBF,
    enumerate_unsafe_regions
)

print("="*80)
print("Enumerating Probable Unsafe Regions")
print("="*80)

# Load data
data_path = '../../data/double_integrator_data.npz'
print(f"\nLoading data from {data_path}...")
data_npz = np.load(data_path)
data = {key: data_npz[key] for key in data_npz.keys()}

# Device
device = 'cpu'
print(f"Using device: {device}")

# Load dynamics model
print("\nLoading models...")
dynamics_hidden_sizes = [32, 32]
f_model = create_mlp(DoubleIntegrator.x_dim + DoubleIntegrator.u_dim, DoubleIntegrator.x_dim, dynamics_hidden_sizes)
f_model_path = '../../models/double_integrator/f.pth'
f_model.load_state_dict(torch.load(f_model_path, map_location=device, weights_only=True))

# Create closed-loop dynamics model
pi_model = DoubleIntegrator.create_policy()
f_pi_model = create_closed_loop_dynamics_model(
    f_model, pi_model, data, DoubleIntegrator.x_low, DoubleIntegrator.x_high, DoubleIntegrator.u_dim
)

# Load constraint model
constraint_hidden_sizes = [16]
h_model = create_mlp(DoubleIntegrator.x_dim, 1, constraint_hidden_sizes)
h_model_path = '../../models/double_integrator/h.pth'
h_model.load_state_dict(torch.load(h_model_path, map_location=device, weights_only=True))

# Load pretrained value function
value_hidden_sizes = [32, 32]
V_model = create_mlp(DoubleIntegrator.x_dim, 1, value_hidden_sizes)
V_pretrain_path = '../../models/double_integrator/V_pretrain.pth'
V_model.load_state_dict(torch.load(V_pretrain_path, map_location=device, weights_only=True))
V_model.eval()

print("✓ All models loaded successfully")

# Create binary decision wrapper
wrapper = BinaryDecisionWrapperCBF(V_model, h_model, f_pi_model, tol=0.0)
input_domain = np.column_stack([DoubleIntegrator.x_low, DoubleIntegrator.x_high])

print("\n" + "="*80)
print("Enumerating Probable Unsafe Regions")
print("="*80)

# Enumerate UNSAFE regions
unsafe_regions, stats = enumerate_unsafe_regions(
    wrapper_model=wrapper,
    input_domain=input_domain,
    n_training_samples=20_000,
    n_decision_trees=1000,
    max_depth=25,
    enumerate_safe_regions=False,  # Enumerate UNSAFE regions
    verify_safe_regions_with_pgd=False,  # No PGD verification for unsafe regions
    pgd_samples=1000,
    pgd_steps=10,
    pgd_eps=0.1,
    wilks_filtering=True,
    wilks_samples=10,
    sample_size_evaluation=10_000,
    desired_coverage=90.0,
    digits_to_consider=0,
    seed=0,
    verbose=True,
    device=device
)

print("\n" + "="*80)
print("Enumeration Results")
print("="*80)
print(f"Number of unsafe regions found: {stats['num_regions']}")
print(f"Coverage of unsafe samples: {stats['coverage']:.2f}%")
print(f"Error rate: {stats['error_rate']:.2f}%")

# Create visualization
print("\n" + "="*80)
print("Creating Visualization")
print("="*80)

# Create grid for value function
n_points = 200
x1 = np.linspace(DoubleIntegrator.x_low[0], DoubleIntegrator.x_high[0], n_points)
x2 = np.linspace(DoubleIntegrator.x_low[1], DoubleIntegrator.x_high[1], n_points)
X1, X2 = np.meshgrid(x1, x2)

# Flatten grid for batch evaluation
X_flat = np.stack([X1.flatten(), X2.flatten()], axis=0).astype(np.float32)

# Compute value function
with torch.no_grad():
    X_torch = torch.from_numpy(X_flat.T)  # [N, 2]
    V_pre = V_model(X_torch).numpy().reshape(n_points, n_points)

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

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Pre-trained Value Function with Enumerated Unsafe Regions
ax = axes[0]
contour = ax.contourf(X1, X2, V_pre, levels=20, cmap='viridis', alpha=0.8)
plt.colorbar(contour, ax=ax, label='Value V(x)')

# Plot true safe region boundary (black line - 100-step forward reachability)
ax.contour(X1, X2, true_safe_region.astype(float), levels=[0.5], colors='black', linewidths=2.5, linestyles='-')

# Plot pre-trained safe region boundary (V=0)
ax.contour(X1, X2, V_pre, levels=[0], colors='blue', linewidths=2.5, linestyles='-')

# Plot enumerated unsafe regions as rectangles
for i, region in enumerate(unsafe_regions):
    # region is shape (n_dims, 2) where [:, 0] is min and [:, 1] is max
    x_min, x_max = region[0, 0], region[0, 1]
    y_min, y_max = region[1, 0], region[1, 1]
    width = x_max - x_min
    height = y_max - y_min
    
    rect = Rectangle((x_min, y_min), width, height, 
                     linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
    ax.add_patch(rect)

ax.set_xlabel('Position', fontsize=12)
ax.set_ylabel('Velocity', fontsize=12)
ax.set_title(f'Pre-trained V(x) with Enumerated Unsafe Regions\n({len(unsafe_regions)} regions found)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', lw=2.5, label='True safe region (100-step)'),
    Line2D([0], [0], color='blue', lw=2.5, label='Pre-trained V=0 boundary'),
    Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='red', alpha=0.3, label=f'Enumerated unsafe regions ({len(unsafe_regions)})')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

# Plot 2: Enumerated Unsafe Regions Coverage Map
ax = axes[1]

# Create binary map of enumerated unsafe regions
from cersyve.enumeration import is_inside
X_flat_samples = X_flat.T  # [N, 2]
inside_enumerated = is_inside(X_flat_samples, unsafe_regions)
inside_map = inside_enumerated.reshape(n_points, n_points).astype(float)

# Create binary map of pre-trained unsafe region
pretrained_unsafe = (V_pre > 0).astype(float)

# Combine into 4-category map:
# 0: Both say safe (white)
# 1: Only pretrained says unsafe (blue)
# 2: Only enumerated says unsafe (red - critical violations)
# 3: Both say unsafe (dark red - verified unsafe)
coverage_map = np.zeros_like(V_pre)
coverage_map[(pretrained_unsafe == 0) & (inside_map == 0)] = 0  # Both safe
coverage_map[(pretrained_unsafe == 1) & (inside_map == 0)] = 1  # Pre-trained only
coverage_map[(pretrained_unsafe == 0) & (inside_map == 1)] = 2  # Enumerated only
coverage_map[(pretrained_unsafe == 1) & (inside_map == 1)] = 3  # Both unsafe

# Create custom colormap
from matplotlib.colors import ListedColormap, BoundaryNorm
colors = ['white', 'blue', 'red', 'darkred']
cmap = ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)

img = ax.imshow(coverage_map, extent=[X1.min(), X1.max(), X2.min(), X2.max()],
                origin='lower', cmap=cmap, norm=norm, aspect='auto')

# Plot true safe region boundary (black line - 100-step forward reachability)
ax.contour(X1, X2, true_safe_region.astype(float), levels=[0.5], colors='black', linewidths=2.5, linestyles='-')

# Add colorbar with labels
cbar = plt.colorbar(img, ax=ax, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(['Both safe', 'Pre-trained only', 'Enumerated only', 'Both unsafe'])

ax.set_xlabel('Position', fontsize=12)
ax.set_ylabel('Velocity', fontsize=12)
ax.set_title(f'Coverage Map: Pre-trained vs Enumerated\n(Error: {stats["error_rate"]:.2f}%)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.suptitle(f'Double Integrator: Enumerated Probable Unsafe Regions\n'
             f'Coverage: {stats["coverage"]:.2f}% | Regions: {len(unsafe_regions)} | Error: {stats["error_rate"]:.2f}%',
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()

# Save figure
save_path = '../../logs/double_integrator/enumerated_unsafe_regions.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {save_path}")

# Calculate statistics
both_unsafe = np.sum(coverage_map == 3)
pretrained_only = np.sum(coverage_map == 1)
enumerated_only = np.sum(coverage_map == 2)
both_safe = np.sum(coverage_map == 0)
total = coverage_map.size

print("\n" + "="*80)
print("Coverage Statistics")
print("="*80)
print(f"Total grid points: {total}")
print(f"Both unsafe (verified): {both_unsafe} ({100*both_unsafe/total:.2f}%)")
print(f"Pre-trained only: {pretrained_only} ({100*pretrained_only/total:.2f}%)")
print(f"Enumerated only (critical violations): {enumerated_only} ({100*enumerated_only/total:.2f}%)")
print(f"Both safe: {both_safe} ({100*both_safe/total:.2f}%)")

# Save enumerated regions for later use
# print("\n" + "="*80)
# print("Saving Enumerated Regions")
# print("="*80)
# regions_save_path = '../../examples/double_integrator/enumerated_unsafe_regions.npz'
# np.savez(regions_save_path, 
#          unsafe_regions=np.array(unsafe_regions, dtype=object),
#          stats=stats,
#          coverage_map=coverage_map,
#          input_domain=input_domain)
# print(f"✓ Regions saved to: {regions_save_path}")

print("\n" + "="*80)
print("Done!")
print("="*80)

plt.savefig("enumeration_results", dpi=300, bbox_inches='tight')
