"""Example: Train constraint model for the double integrator system."""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from systems.double_integrator import DoubleIntegrator
from cersyve import create_mlp, train_constraint

# Create constraint model
constraint_hidden_sizes = [16]
h_model = create_mlp(DoubleIntegrator.x_dim, 1, constraint_hidden_sizes)

print(f"Constraint model architecture:")
print(f"  Input dim: {DoubleIntegrator.x_dim}")
print(f"  Hidden sizes: {constraint_hidden_sizes}")
print(f"  Output dim: 1")
print(f"  Total parameters: {sum(p.numel() for p in h_model.parameters())}")

# Train constraint model
print("\nTraining constraint model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set random seed for reproducibility
seed = 42

log_path = train_constraint(
    x_low=DoubleIntegrator.x_low,
    x_high=DoubleIntegrator.x_high,
    h_model=h_model,
    constraint=DoubleIntegrator.constraint,
    lr=3e-4,
    batch_size=256,
    n_iter=100000,
    log_dir='../../logs/double_integrator/',
    device=device,
    seed=seed
)

print(f"\nTraining complete!")
print(f"Model saved to: models/double_integrator/h.pth")
print(f"Logs saved to: {log_path}")
print(f"\nTo view training curves, run:")
print(f"  tensorboard --logdir={log_path}")
