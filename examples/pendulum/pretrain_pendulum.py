"""Example: Pretrain value function for the pendulum system."""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from systems.pendulum import Pendulum
from cersyve import create_mlp, create_closed_loop_dynamics_model, pretrain_value

# Load data
data_path = '../../data/pendulum_data.npz'
print(f"Loading data from {data_path}...")
data_npz = np.load(data_path)
data = {key: data_npz[key] for key in data_npz.keys()}

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load dynamics model
dynamics_hidden_sizes = [32, 32]
f_model = create_mlp(Pendulum.x_dim + Pendulum.u_dim, Pendulum.x_dim, dynamics_hidden_sizes)
f_model_path = '../../models/pendulum/f.pth'
print(f"Loading dynamics model from {f_model_path}...")
f_model.load_state_dict(torch.load(f_model_path, map_location=device, weights_only=True))

# Create closed-loop dynamics model
pi_model = Pendulum.create_policy()
f_pi_model = create_closed_loop_dynamics_model(
    f_model, pi_model, data, Pendulum.x_low, Pendulum.x_high, Pendulum.u_dim
)
print("Created closed-loop dynamics model")

# Load constraint model
constraint_hidden_sizes = [16]
h_model = create_mlp(Pendulum.x_dim, 1, constraint_hidden_sizes)
h_model_path = '../../models/pendulum/h.pth'
print(f"Loading constraint model from {h_model_path}...")
h_model.load_state_dict(torch.load(h_model_path, map_location=device, weights_only=True))

# Create value function model
value_hidden_sizes = [32, 32]
V_model = create_mlp(Pendulum.x_dim, 1, value_hidden_sizes)

print(f"\nValue function model architecture:")
print(f"  Input dim: {Pendulum.x_dim}")
print(f"  Hidden sizes: {value_hidden_sizes}")
print(f"  Output dim: 1")
print(f"  Total parameters: {sum(p.numel() for p in V_model.parameters())}")

# Pretrain value function
print("\nPretraining value function...")

# Set random seed for reproducibility
seed = 42

log_path = pretrain_value(
    V_model=V_model,
    f_pi_model=f_pi_model,
    h_model=h_model,
    x_low=Pendulum.x_low,
    x_high=Pendulum.x_high,
    gamma=0.9,
    lr=3e-4,
    batch_size=256,
    iter_num=100000,
    weight_decay=0.0,
    penalty="APA",
    noise_scale=0.1,
    space_size=Pendulum.x_high - Pendulum.x_low,
    apa_coef=1e-4,
    log_dir='../../logs/pendulum/',
    device=device,
    seed=seed,
    task="pendulum",
)

print(f"\nPretraining complete!")
print(f"Model saved to: models/pendulum/V_pretrain.pth")
print(f"Logs saved to: {log_path}")
print(f"\nTo view training curves, run:")
print(f"  tensorboard --logdir={log_path}")
