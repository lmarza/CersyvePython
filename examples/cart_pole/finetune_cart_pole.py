"""
Finetune V for CartPole (PGD/BGB search, no MILP).
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from systems.cart_pole import CartPole
from cersyve import create_mlp, create_closed_loop_dynamics_model, finetune_value

def main():
    data_path = "../../data/cart_pole_data.npz"
    models_dir = "../../models/cart_pole/"
    logs_dir = "../../logs/cart_pole/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_npz = np.load(data_path)
    data = {k: data_npz[k] for k in data_npz.keys()}

    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_model.load_state_dict(torch.load(os.path.join(models_dir, "f.pth"), map_location=device))
    f_model.to(device).eval()

    pi_model = CartPole.create_policy().to(device).eval()
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, CartPole.x_low, CartPole.x_high, CartPole.u_dim
    ).to(device).eval()

    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_model.load_state_dict(torch.load(os.path.join(models_dir, "h.pth"), map_location=device))
    h_model.to(device).eval()

    V_model = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_model.load_state_dict(torch.load(os.path.join(models_dir, "V_pretrain.pth"), map_location=device))
    V_model.to(device).train()

    log_path = finetune_value(
        V_model=V_model,
        f_pi_model=f_pi_model,
        h_model=h_model,
        x_low=CartPole.x_low,
        x_high=CartPole.x_high,
        lr=1e-4,
        max_iter=200000,
        search_size=8000,
        bnd_eps=0.25,
        tol=1e-3,
        search_method="BGB",
        pgd_step=10,
        pgd_eps=0.1,
        backtrack_step=20,
        length_discount=0.8,
        direct_discount=0.5,
        capacity=20000,
        sample_size=256,
        search_stop=5000,
        replay=1,
        max_skip=1000,
        reg_method="ESR",
        reg_coef=0.1,
        log_dir=logs_dir,
        eval_every=50,
        save_every=2000,
        device=device,
        seed=42
    )
    print("Logs:", log_path)

if __name__ == "__main__":
    main()