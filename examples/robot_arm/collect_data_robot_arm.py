"""
Example: Collect trajectory data for the RobotArm system.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from systems.robot_arm import RobotArm
from cersyve import collect_data

# Set parameters
seed = 42  # Random seed for reproducibility
step_num = 1_000_000
max_ep_len = 200
noise_scale = 0.01
save_path = '../../data/robot_arm_data.npz'

print("Collecting data for RobotArm system...")
print(f"  Total steps: {step_num}")
print(f"  Max episode length: {max_ep_len}")
print(f"  State space: [{RobotArm.x_low}, {RobotArm.x_high}]")
print(f"  Action space: [{RobotArm.u_low}, {RobotArm.u_high}]")
print(f"  Number of joints: {RobotArm.n}")
print(f"  Time step: {RobotArm.dt}")

# Collect data
data = collect_data(
    x_low=RobotArm.x_low,
    x_high=RobotArm.x_high,
    u_low=RobotArm.u_low,
    u_high=RobotArm.u_high,
    dynamics=RobotArm.dynamics,
    terminated=RobotArm.terminated,
    step_num=step_num,
    max_ep_len=max_ep_len,
    noise_scale=noise_scale,
    save_path=save_path,
    seed=seed
)

print(f"\nData saved to {save_path}")
print("Data shapes:")
print(f"  x: {data['x'].shape}")
print(f"  u: {data['u'].shape}")
print(f"  dx: {data['dx'].shape}")

print("\nData statistics:")
print(f"  x_mean: {data['x_mean']}")
print(f"  x_std:  {data['x_std']}")
print(f"  u_mean: {data['u_mean']}")
print(f"  u_std:  {data['u_std']}")
print(f"  dx_mean:{data['dx_mean']}")
print(f"  dx_std: {data['dx_std']}")