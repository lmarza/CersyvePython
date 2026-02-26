"""Example: Finetune value function for the double integrator system."""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from systems.double_integrator import DoubleIntegrator
from cersyve import create_mlp, create_closed_loop_dynamics_model, finetune_value

# Load data
data_path = '../../data/double_integrator_data.npz'
print(f"Loading data from {data_path}...")
data_npz = np.load(data_path)
data = {key: data_npz[key] for key in data_npz.keys()}

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load dynamics model
dynamics_hidden_sizes = [32, 32]
f_model = create_mlp(DoubleIntegrator.x_dim + DoubleIntegrator.u_dim, DoubleIntegrator.x_dim, dynamics_hidden_sizes)
f_model_path = '../../models/double_integrator/f.pth'
print(f"Loading dynamics model from {f_model_path}...")
f_model.load_state_dict(torch.load(f_model_path, map_location=device, weights_only=True))

# Create closed-loop dynamics model
pi_model = DoubleIntegrator.create_policy()
f_pi_model = create_closed_loop_dynamics_model(
    f_model, pi_model, data, DoubleIntegrator.x_low, DoubleIntegrator.x_high, DoubleIntegrator.u_dim
)
print("Created closed-loop dynamics model")

# Load constraint model
constraint_hidden_sizes = [16]
h_model = create_mlp(DoubleIntegrator.x_dim, 1, constraint_hidden_sizes)
h_model_path = '../../models/double_integrator/h.pth'  # Update path as needed
print(f"Loading constraint model from {h_model_path}...")
h_model.load_state_dict(torch.load(h_model_path, map_location=device, weights_only=True))

# Create and load pretrained value function
value_hidden_sizes = [32, 32]
V_model = create_mlp(DoubleIntegrator.x_dim, 1, value_hidden_sizes)
V_pretrain_path = '../../models/double_integrator/V_pretrain.pth'  # Update path as needed
print(f"Loading pretrained value function from {V_pretrain_path}...")
V_model.load_state_dict(torch.load(V_pretrain_path, map_location=device, weights_only=True))

print(f"\nValue function model architecture:")
print(f"  Input dim: {DoubleIntegrator.x_dim}")
print(f"  Hidden sizes: {value_hidden_sizes}")
print(f"  Output dim: 1")
print(f"  Total parameters: {sum(p.numel() for p in V_model.parameters())}")

# Finetune value function using counterexample-guided search
print("\nFinetuning value function...")

# Set random seed for reproducibility
seed = 42

log_path = finetune_value(
    V_model=V_model,
    f_pi_model=f_pi_model,
    h_model=h_model,
    x_low=DoubleIntegrator.x_low,
    x_high=DoubleIntegrator.x_high,
    lr=1e-4,
    max_iter=100000,
    search_size=200,
    bnd_ratio=0.1,
    bnd_ratio_avg=0.9,
    min_bnd_ratio=0.01,
    max_bnd_ratio=1.0,
    bnd_eps=0.1,
    search_method="BGB",  # Boundary-Guided Branching
    pgd_step=10,
    pgd_eps=0.1,
    backtrack_step=20,
    length_discount=0.8,
    direct_discount=0.5,
    tol=1e-6,
    capacity=10000,
    sample_size=100,
    search_stop=1000,
    replay=1,
    max_skip=200,
    reg_method="ESR",  # Entering State Regularization
    eps_h=0.01,
    eps_v=0.01,
    reg_coef=0.1,
    log_dir='../../logs/double_integrator/',
    eval_every=10,
    save_every=1000,
    device=device,
    seed=seed
)

print(f"\nFinetuning complete!")
print(f"Model saved to: models/double_integrator/V_finetune.pth")
print(f"Logs saved to: {log_path}")
print(f"\nTo view training curves, run:")
print(f"  tensorboard --logdir={log_path}")
