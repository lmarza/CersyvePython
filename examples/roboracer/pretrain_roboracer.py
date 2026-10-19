"""Pretrain the value function of the RoboRacer tube benchmark (RoboRacer)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from systems.roboracer import RoboRacer
from cersyve import create_mlp, pretrain_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system = RoboRacer
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    model_dir = os.path.join(root, "models/roboracer")
    os.makedirs(model_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    f_pi_model = system.create_closed_loop_dynamics_model()
    h_model = system.create_constraint_model()
    V_model = create_mlp(system.x_dim, 1, [32, 32])

    pretrain_value(
        V_model=V_model,
        f_pi_model=f_pi_model,
        h_model=h_model,
        x_low=system.x_low,
        x_high=system.x_high,
        gamma=0.9,
        lr=3e-4,
        batch_size=256,
        iter_num=100000,
        weight_decay=0.0,
        penalty="APA",
        noise_scale=0.1,
        space_size=system.x_high - system.x_low,
        apa_coef=1e-4,
        log_dir=os.path.join(model_dir, "logs") + os.sep,
        device=device,
        seed=args.seed,
        save_path=os.path.join(model_dir, "V_pretrain.pth"),
    )


if __name__ == "__main__":
    main()
