"""
Collect trajectory data for CartPole.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from systems.cart_pole import CartPole
from cersyve import collect_data

def main():
    seed = 42
    step_num = 1_000_000
    max_ep_len = 200
    noise_scale = 0.01

    save_path = "../../data/cart_pole_data.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Collecting CartPole data...")
    data = collect_data(
        x_low=CartPole.x_low,
        x_high=CartPole.x_high,
        u_low=CartPole.u_low,
        u_high=CartPole.u_high,
        dynamics=CartPole.dynamics,
        terminated=CartPole.terminated,
        step_num=step_num,
        max_ep_len=max_ep_len,
        noise_scale=noise_scale,
        save_path=save_path,
        seed=seed
    )

    print("Saved:", save_path)
    print("x,u,dx shapes:", data["x"].shape, data["u"].shape, data["dx"].shape)
    print("x_std:", data["x_std"], "u_std:", data["u_std"], "dx_std:", data["dx_std"])

if __name__ == "__main__":
    main()