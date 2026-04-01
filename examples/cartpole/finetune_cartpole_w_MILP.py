"""Example: Finetune value function for the cartpole system using MILP."""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cersyve.finetune import finetune_value_w_MILP

import torch
import numpy as np
from systems.cartpole import CartPole
from cersyve import create_mlp, create_closed_loop_dynamics_model, finetune_value_w_MILP

# Load data
data_path = '../../data/cartpole_data.npz'
print(f"Loading data from {data_path}...")
data_npz = np.load(data_path)
data = {key: data_npz[key].astype(np.float32) for key in data_npz.keys()}

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load dynamics model
dynamics_hidden_sizes = [32, 32]
f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, dynamics_hidden_sizes)
f_model = f_model.to(device)
f_model_path = '../../models/cartpole/f.pth'
print(f"Loading dynamics model from {f_model_path}...")
f_model.load_state_dict(torch.load(f_model_path, map_location=device, weights_only=True))

# Create closed-loop dynamics model
pi_model = CartPole.create_policy()
pi_model = pi_model.float().to(device)
f_pi_model = create_closed_loop_dynamics_model(
    f_model, pi_model, data, CartPole.x_low, CartPole.x_high, CartPole.u_dim
)
print("Created closed-loop dynamics model")

# Load constraint model
constraint_hidden_sizes = [16]
h_model = create_mlp(CartPole.x_dim, 1, constraint_hidden_sizes)
h_model = h_model.to(device)
h_model_path = '../../models/cartpole/h.pth'
print(f"Loading constraint model from {h_model_path}...")
h_model.load_state_dict(torch.load(h_model_path, map_location=device, weights_only=True))

# Create and load pretrained value function
# print(f"\nValue function model architecture:")
# print(f"  Input dim: {CartPole.x_dim}")
# print(f"  Hidden sizes: {value_hidden_sizes}")
# print(f"  Output dim: 1")
# print(f"  Total parameters: {sum(p.numel() for p in V_model.parameters())}")

# Finetune value function using counterexample-guided search
print("\nFinetuning value function...")

# Set random seed for reproducibility
#seeds = [42, 2101, 70]
seeds = [1, 0, 2]
tfrs = []
iters = []

for seed in seeds:
    print("Test seed:", seed)
    value_hidden_sizes = [32, 32]
    V_model = create_mlp(CartPole.x_dim, 1, value_hidden_sizes)
    V_model = V_model.to(device)
    V_pretrain_path = '../../models/cartpole/V_pretrain.pth'
    print(f"Loading pretrained value function from {V_pretrain_path}...")
    V_model.load_state_dict(torch.load(V_pretrain_path, map_location=device, weights_only=True))
    
    log_path, tfr, iterations = finetune_value_w_MILP(
        V_model=V_model,
        f_pi_model=f_pi_model,
        h_model=h_model,
        x_low=CartPole.x_low.astype(np.float32),
        x_high=CartPole.x_high.astype(np.float32),
        lr=1e-4,
        max_iter=100000,
        search_size=1000,
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
        tol=1e-4,
        capacity=10000,
        sample_size=100,
        search_stop=1000,
        replay=1,
        max_skip=1000,
        reg_method="ESR",  # Entering State Regularization
        eps_h=0.01,
        eps_v=0.01,
        reg_coef=0.1,
        log_dir='../../logs/cartpole/',
        eval_every=10,
        save_every=1000,
        milp_timeout=500,
        device=device,
        seed=seed,
        task="cartpole"
    )

    tfrs.append(tfr)
    iters.append(iterations)


print(f"\nFinetuning complete!")
print(f"Model saved to: models/cartpole/V_finetune_milp.pth")
print(f"Logs saved to: {log_path}")
print(f"\nTo view training curves, run:")
print(f"  tensorboard --logdir={log_path}")
print(f"Mean True Feasibility Region (TFR): {np.mean(tfrs)*100:.4f}% (std: {np.std(tfrs)*100:.4f})")
print(f"Mean iterations: {np.mean(iters):.2f} (std: {np.std(iters):.2f})")
