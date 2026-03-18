"""
Pretrain value function V for CartPole.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from systems.cart_pole import CartPole
from cersyve import create_mlp, create_closed_loop_dynamics_model, pretrain_value

def main():
    data_path = "../../data/cart_pole_data.npz"
    models_dir = "../../models/cart_pole/"
    logs_dir = "../../logs/cart_pole/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_npz = np.load(data_path)
    data = {k: data_npz[k] for k in data_npz.keys()}

    # f_model
    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_model.load_state_dict(torch.load(os.path.join(models_dir, "f.pth"), map_location=device))
    f_model.to(device).eval()

    # pi_model (LQR + clip)
    pi_model = CartPole.create_policy().to(device).eval()

    # closed-loop learned model
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, CartPole.x_low, CartPole.x_high, CartPole.u_dim
    ).to(device).eval()

    # h_model
    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_model.load_state_dict(torch.load(os.path.join(models_dir, "h.pth"), map_location=device))
    h_model.to(device).eval()

    # V_model
    V_model = create_mlp(CartPole.x_dim, 1, [64, 64]).to(device).train()

    log_path = pretrain_value(
        V_model=V_model,
        f_pi_model=f_pi_model,
        h_model=h_model,
        x_low=CartPole.x_low,
        x_high=CartPole.x_high,
        gamma=0.9,
        lr=3e-4,
        batch_size=512,
        iter_num=150_000,
        weight_decay=0.0,
        penalty="APA",
        noise_scale=0.1,
        space_size=(CartPole.x_high - CartPole.x_low),
        apa_coef=1e-3,
        log_dir=logs_dir,
        device=device,
        seed=42
    )
    print("Logs:", log_path)
    print("Saved to:", os.path.join(models_dir, "V_pretrain.pth"))

if __name__ == "__main__":
    main()