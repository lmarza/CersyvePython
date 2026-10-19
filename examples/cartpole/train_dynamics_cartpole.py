"""Example: Train dynamics model for the cartpole system."""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from systems.cartpole import CartPole
from cersyve import create_mlp, train_dynamics

# Load data
data_path = '../../data/cartpole_data.npz'
print(f"Loading data from {data_path}...")
data_npz = np.load(data_path)
data = {key: data_npz[key] for key in data_npz.keys()}

# Create dynamics model
dynamics_hidden_sizes = [32, 32]
f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, dynamics_hidden_sizes)

print(f"\nDynamics model architecture:")
print(f"  Input dim: {CartPole.x_dim + CartPole.u_dim}")
print(f"  Hidden sizes: {dynamics_hidden_sizes}")
print(f"  Output dim: {CartPole.x_dim}")
print(f"  Total parameters: {sum(p.numel() for p in f_model.parameters())}")

# Train dynamics model
print("\nTraining dynamics model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set random seed for reproducibility
seed = 42

# Combine state and action space sizes for noise scaling
xu_space_size = np.concatenate([
    (CartPole.x_high - CartPole.x_low).astype(np.float32),
    (CartPole.u_high - CartPole.u_low).astype(np.float32)
])

log_path = train_dynamics(
    data=data,
    f_model=f_model,
    closed_loop=False,
    lr=1e-3,
    batch_size=256,
    epoch_num=100,
    weight_decay=0.0,
    penalty="APA",
    noise_scale=0.1,
    space_size=xu_space_size,
    apa_coef=1e-4,
    log_dir='../../logs/cartpole/',
    device=device,
    seed=seed,
    task="cartpole"
)

print(f"\nTraining complete!")
print(f"Model saved to: models/cartpole/f.pth")
print(f"Logs saved to: {log_path}")
print(f"\nTo view training curves, run:")
print(f"  tensorboard --logdir={log_path}")
