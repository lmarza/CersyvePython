"""
Train dynamics model f(x,u)->dx for CartPole.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from systems.cart_pole import CartPole
from cersyve import create_mlp, train_dynamics

def main():
    data_path = "../../data/cart_pole_data.npz"
    models_dir = "../../models/cart_pole/"
    logs_dir = "../../logs/cart_pole/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    data_npz = np.load(data_path)
    data = {k: data_npz[k] for k in data_npz.keys()}

    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xu_space_size = np.concatenate([CartPole.x_high - CartPole.x_low, CartPole.u_high - CartPole.u_low])

    log_path = train_dynamics(
        data=data,
        f_model=f_model,
        closed_loop=False,
        lr=1e-3,
        batch_size=256,
        epoch_num=150,
        weight_decay=0.0,
        penalty="APA",
        noise_scale=0.1,
        space_size=xu_space_size,
        apa_coef=1e-3,
        log_dir=logs_dir,
        device=device,
        seed=42
    )
    print("Logs:", log_path)
    print("Saved model to:", os.path.join(models_dir, "f.pth"))

if __name__ == "__main__":
    main()