"""
Finetune V for CartPole (Enumeration + MILP).
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from systems.cart_pole import CartPole
from cersyve import create_mlp, create_closed_loop_dynamics_model
from cersyve.finetune import finetune_value_enum_MILP

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

    save_path = os.path.join(models_dir, "V_finetune_enum_milp.pth")

    log_path = finetune_value_enum_MILP(
        V_model=V_model,
        f_pi_model=f_pi_model,
        h_model=h_model,
        x_low=CartPole.x_low,
        x_high=CartPole.x_high,
        lr=1e-4,
        max_iter=200000,
        tol=1e-4,

        capacity=20000,
        sample_size=256,
        replay=1,
        max_skip=500,

        # Enumeration: 4D is heavier than 2D → start moderate
        enum_freq=2000,
        enum_n_samples=30000,
        enum_coverage=70.0,
        enum_n_trees=800,
        enum_max_depth=14,
        enum_n_train=60000,
        refill_threshold=1000,

        # Fallback
        fallback_eval_rounds=3,
        fallback_eval_samples=60000,
        fallback_n_neighbors=8000,
        fallback_radius=0.05,

        # Reg
        search_size=4000,
        reg_method="ESR",
        reg_coef=0.1,

        # MILP
        milp_timeout=300,

        log_dir=logs_dir,
        eval_every=100,
        save_every=5000,
        save_path=save_path,
        device=device,
        seed=42
    )
    print("Logs:", log_path)
    print("Saved:", save_path)

if __name__ == "__main__":
    main()