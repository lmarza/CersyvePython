"""Example: Collect trajectory data for the cartpole system."""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from systems.cartpole import CartPole
from cersyve import collect_data

# Set parameters
seed = 42  # Random seed for reproducibility
step_num = 1000000
max_ep_len = 100
noise_scale = 0.01
save_path = '../../data/cartpole_data.npz'

print("Collecting data for cartpole system...")
print(f"  Total steps: {step_num}")
print(f"  Max episode length: {max_ep_len}")
print(f"  State space: [{CartPole.x_low}, {CartPole.x_high}]")
print(f"  Action space: [{CartPole.u_low}, {CartPole.u_high}]")

# Collect data
data = collect_data(
    x_low=CartPole.x_low.astype(np.float32),
    x_high=CartPole.x_high.astype(np.float32),
    u_low=CartPole.u_low.astype(np.float32),
    u_high=CartPole.u_high.astype(np.float32),
    dynamics=CartPole.dynamics,
    terminated=CartPole.terminated,
    step_num=step_num,
    max_ep_len=max_ep_len,
    noise_scale=noise_scale,
    save_path=save_path,
    seed=seed
)

print(f"\nData saved to {save_path}")
print(f"Data shapes:")
print(f"  x: {data['x'].shape}")
print(f"  u: {data['u'].shape}")
print(f"  dx: {data['dx'].shape}")
print("\nData statistics:")
print(f"  x_mean: {data['x_mean']}")
print(f"  x_std: {data['x_std']}")
print(f"  u_mean: {data['u_mean']}")
print(f"  u_std: {data['u_std']}")
print(f"  dx_mean: {data['dx_mean']}")
print(f"  dx_std: {data['dx_std']}")
