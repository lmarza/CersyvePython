"""
Finetune V for CartPole (sampling-based + MILP) — ALL FLOAT32 + FORCED SAVE artifacts.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from systems.cart_pole import CartPole
from cersyve import create_mlp, create_closed_loop_dynamics_model
from cersyve.finetune import finetune_value_w_MILP


# -----------------------------
# Robust saving utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def atomic_torch_save(state_dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)


def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


print("RUNNING FILE:", os.path.abspath(__file__))


def main():
    # -----------------------------
    # Config
    # -----------------------------
    print("SYNC TEST 123")
    print("AUTO UPLOAD OK")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    data_path = os.path.join(base_dir, "data", "cart_pole_data.npz")
    f_path = os.path.join(base_dir, "models", "cart_pole", "f.pth")
    h_path = os.path.join(base_dir, "models", "cart_pole", "h.pth")
    V_pretrain_path = os.path.join(base_dir, "models", "cart_pole", "V_pretrain.pth")

    model_dir = os.path.join(base_dir, "models", "cart_pole")
    log_dir = os.path.join(base_dir, "logs", "cart_pole")
    ensure_dir(model_dir)
    ensure_dir(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    V_snapshot_path = os.path.join(model_dir, f"V_finetuned_milp_{timestamp}.pth")
    summary_path = os.path.join(model_dir, f"V_finetuned_milp_{timestamp}.json")
    pretrain_snapshot_path = os.path.join(model_dir, "V_pretrain_snapshot_tmp.pth")
    final_model_path = os.path.join(model_dir, "V_finetune_milp.pth")

    # -----------------------------
    # Load data
    # -----------------------------
    print(f"Loading data from {data_path}...")
    data_npz = np.load(data_path)
    data = {key: data_npz[key] for key in data_npz.keys()}

    # -----------------------------
    # Build models
    # -----------------------------
    pi = CartPole.create_policy().to(device).float()

    # Hidden sizes aligned with saved checkpoints
    dynamics_hidden_sizes = [64, 64]    # matches f.pth
    constraint_hidden_sizes = [32, 32]  # matches h.pth
    value_hidden_sizes = [64, 64]       # matches V_pretrain.pth

    # create_mlp(input_dim, output_dim, hidden_sizes)
    f = create_mlp(
        CartPole.x_dim + CartPole.u_dim,
        CartPole.x_dim,
        dynamics_hidden_sizes
    ).to(device).float()

    h = create_mlp(
        CartPole.x_dim,
        1,
        constraint_hidden_sizes
    ).to(device).float()

    V = create_mlp(
        CartPole.x_dim,
        1,
        value_hidden_sizes
    ).to(device).float()

    # -----------------------------
    # Load weights
    # -----------------------------
    print(f"Loading f from {f_path}...")
    f.load_state_dict(torch.load(f_path, map_location=device, weights_only=True))

    print(f"Loading h from {h_path}...")
    h.load_state_dict(torch.load(h_path, map_location=device, weights_only=True))

    print(f"Loading V_pretrain from {V_pretrain_path}...")
    V.load_state_dict(torch.load(V_pretrain_path, map_location=device, weights_only=True))

    print("pi dtype:", next(pi.parameters()).dtype)
    print("f dtype:", next(f.parameters()).dtype)
    print("h dtype:", next(h.parameters()).dtype)
    print("V dtype:", next(V.parameters()).dtype)

    # -----------------------------
    # Create closed-loop dynamics
    # -----------------------------
    print("Building closed-loop dynamics model...")
    try:
        f_pi = create_closed_loop_dynamics_model(f, pi)
    except TypeError:
        f_pi = create_closed_loop_dynamics_model(
            f, pi, data, CartPole.x_low, CartPole.x_high, CartPole.u_dim
        )

    # -----------------------------
    # Save snapshot of pretrain
    # -----------------------------
    atomic_torch_save(V.state_dict(), pretrain_snapshot_path)
    print(f"Saved pretrain snapshot: {pretrain_snapshot_path}")

    # -----------------------------
    # Finetuning
    # -----------------------------
    print("\nFinetuning (sampling-based + MILP)...")
    print("Using safer settings to reduce repeated/inconclusive MILP calls:")
    print("  max_skip = 5000")
    print("  milp_timeout = 180")
    print(f"  final save path = {final_model_path}")

    log_path, tfr, iterations = finetune_value_w_MILP(
        V_model=V,
        f_pi_model=f_pi,
        h_model=h,
        x_low=CartPole.x_low.astype(np.float32),
        x_high=CartPole.x_high.astype(np.float32),
        lr=1e-4,
        max_iter=200000,
        search_size=1000,
        bnd_ratio=0.1,
        bnd_ratio_avg=0.9,
        min_bnd_ratio=0.01,
        max_bnd_ratio=1.0,
        bnd_eps=0.1,
        search_method="BGB",
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
        max_skip=5000,          # was 1000
        reg_method="ESR",
        eps_h=0.01,
        eps_v=0.01,
        reg_coef=0.1,
        milp_timeout=180,       # was 300
        log_dir=log_dir,
        eval_every=10,
        save_every=1000,
        save_path=final_model_path,
        device=device,
        seed=42,
    )

    print("\nReturn from finetune_value_w_MILP:")
    print("log_path:", log_path)
    print("tfr:", tfr)
    print("iterations:", iterations)

    # -----------------------------
    # Extra timestamped snapshot
    # -----------------------------
    atomic_torch_save(V.state_dict(), V_snapshot_path)
    print(f"Saved finetuned V snapshot to: {V_snapshot_path}")

    # -----------------------------
    # Save summary metadata
    # -----------------------------
    summary = {
        "system": "cart_pole",
        "timestamp": timestamp,
        "device": device,
        "data_path": os.path.abspath(data_path),
        "f_path": os.path.abspath(f_path),
        "h_path": os.path.abspath(h_path),
        "V_pretrain_path": os.path.abspath(V_pretrain_path),
        "pretrain_snapshot_path": os.path.abspath(pretrain_snapshot_path),
        "final_model_path": os.path.abspath(final_model_path),
        "timestamped_model_path": os.path.abspath(V_snapshot_path),
        "log_path": os.path.abspath(log_path),
        "tfr": float(tfr),
        "iterations": int(iterations),
        "used_saved_controller": True,
        "policy_source": "CartPole.create_policy()",
        "finetune_config": {
            "lr": 1e-4,
            "max_iter": 200000,
            "search_size": 1000,
            "bnd_ratio": 0.1,
            "bnd_ratio_avg": 0.9,
            "min_bnd_ratio": 0.01,
            "max_bnd_ratio": 1.0,
            "bnd_eps": 0.1,
            "search_method": "BGB",
            "pgd_step": 10,
            "pgd_eps": 0.1,
            "backtrack_step": 20,
            "length_discount": 0.8,
            "direct_discount": 0.5,
            "tol": 1e-4,
            "capacity": 10000,
            "sample_size": 100,
            "search_stop": 1000,
            "replay": 1,
            "max_skip": 5000,
            "reg_method": "ESR",
            "eps_h": 0.01,
            "eps_v": 0.01,
            "reg_coef": 0.1,
            "milp_timeout": 180,
            "eval_every": 10,
            "save_every": 1000,
            "seed": 42,
        },
    }

    save_json(summary, summary_path)
    print(f"Saved summary to: {summary_path}")

    print("\nTraining complete.")
    print(f"Final model: {final_model_path}")
    print(f"Timestamped snapshot: {V_snapshot_path}")
    print(f"Logs: {log_path}")
    print(f"TFR: {tfr:.6f}")
    print(f"Last iteration: {iterations}")


if __name__ == "__main__":
    main()