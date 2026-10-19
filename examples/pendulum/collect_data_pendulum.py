"""Example: Collect trajectory data for the pendulum system."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from systems.pendulum import Pendulum
from cersyve import collect_data

# Set parameters
seed = 42  # Random seed for reproducibility
step_num = 1000000
max_ep_len = 100
noise_scale = 0.01
save_path = '../../data/pendulum_data.npz'

print("Collecting data for pendulum system...")
print(f"  Total steps: {step_num}")
print(f"  Max episode length: {max_ep_len}")
print(f"  State space: [{Pendulum.x_low}, {Pendulum.x_high}]")
print(f"  Action space: [{Pendulum.u_low}, {Pendulum.u_high}]")

# Collect data
data = collect_data(
    x_low=Pendulum.x_low,
    x_high=Pendulum.x_high,
    u_low=Pendulum.u_low,
    u_high=Pendulum.u_high,
    dynamics=Pendulum.dynamics,
    terminated=Pendulum.terminated,
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
