"""
Train constraint model h(x) for CartPole using the analytic constraint.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from systems.cart_pole import CartPole
from cersyve import create_mlp, train_constraint

def main():
    models_dir = "../../models/cart_pole/"
    logs_dir = "../../logs/cart_pole/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = train_constraint(
        x_low=CartPole.x_low,
        x_high=CartPole.x_high,
        h_model=h_model,
        constraint=CartPole.constraint,
        lr=3e-4,
        batch_size=256,
        n_iter=150_000,
        log_dir=logs_dir,
        device=device,
        seed=42
    )
    print("Logs:", log_path)
    print("Saved model to:", os.path.join(models_dir, "h.pth"))

if __name__ == "__main__":
    main()