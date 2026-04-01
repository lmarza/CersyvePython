"""
Finetune V for CartPole (sampling-based + MILP) — multi-seed + robust save + JSON summaries.
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
    pretrain_snapshot_path = os.path.join(model_dir, "V_pretrain_snapshot_tmp.pth")
    aggregate_summary_path = os.path.join(model_dir, f"V_finetuned_milp_aggregate_{timestamp}.json")

    # -----------------------------
    # Load data
    # -----------------------------
    print(f"Loading data from {data_path}...")
    data_npz = np.load(data_path)
    data = {key: data_npz[key] for key in data_npz.keys()}

    # -----------------------------
    # Build shared models
    # -----------------------------
    pi = CartPole.create_policy().to(device).float()

    # Hidden sizes aligned with saved checkpoints
    dynamics_hidden_sizes = [64, 64]    # matches f.pth
    constraint_hidden_sizes = [32, 32]  # matches h.pth
    value_hidden_sizes = [64, 64]       # matches V_pretrain.pth

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

    # -----------------------------
    # Load shared weights
    # -----------------------------
    print(f"Loading f from {f_path}...")
    f.load_state_dict(torch.load(f_path, map_location=device, weights_only=True))

    print(f"Loading h from {h_path}...")
    h.load_state_dict(torch.load(h_path, map_location=device, weights_only=True))

    print("pi dtype:", next(pi.parameters()).dtype)
    print("f dtype:", next(f.parameters()).dtype)
    print("h dtype:", next(h.parameters()).dtype)

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
    V_tmp = create_mlp(
        CartPole.x_dim,
        1,
        value_hidden_sizes
    ).to(device).float()

    print(f"Loading V_pretrain from {V_pretrain_path}...")
    V_tmp.load_state_dict(torch.load(V_pretrain_path, map_location=device, weights_only=True))

    atomic_torch_save(V_tmp.state_dict(), pretrain_snapshot_path)
    print(f"Saved pretrain snapshot: {pretrain_snapshot_path}")

    # -----------------------------
    # Finetuning config
    # -----------------------------
    # conf commetata arriva a 6 minuti e 10 secondi totale con 2 minuti e 20 di train e il restante di verifica e tfr = 0,649
    finetune_config = {
        "lr": 1e-4,
        "max_iter": 10000, #paper 100000 io 4000
        "search_size": 100, #paper 1000
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
        "capacity": 1000,
        "sample_size": 50, # mio 10
        "search_stop": 100,
        "replay": 1,
        "max_skip": 50, #50
        "reg_method": "ESR",
        "eps_h": 0.01,
        "eps_v": 0.01,
        "reg_coef": 0.1,
        "milp_timeout": 90, #120
        "milp_start_iter": 1200, #1500
        "eval_every": 10,
        "save_every": 1000,
        "milp_box_shrink": 1.0,
        "milp_box_grow": 1.0,
        "milp_box_max": 1.0,
        "milp_max_inconclusive": 1,
    }

    print("\nFinetuning (sampling-based + MILP)...")
    print("Using settings:")
    print(f"  max_skip = {finetune_config['max_skip']}")
    print(f"  milp_timeout = {finetune_config['milp_timeout']}")
    print(f"  milp_start_iter = {finetune_config['milp_start_iter']}")

    # -----------------------------
    # Multi-seed runs
    # -----------------------------
    seeds = [1, 0, 2]
    tfrs = []
    iters = []
    run_summaries = []

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"RUNNING SEED {seed}")
        print("=" * 70)

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_final_model_path = os.path.join(model_dir, f"V_finetune_milp_seed{seed}.pth")
        run_snapshot_path = os.path.join(model_dir, f"V_finetuned_milp_seed{seed}_{run_timestamp}.pth")
        run_summary_path = os.path.join(model_dir, f"V_finetuned_milp_seed{seed}_{run_timestamp}.json")

        # Rebuild V from the same pretrained checkpoint for every seed
        V = create_mlp(
            CartPole.x_dim,
            1,
            value_hidden_sizes
        ).to(device).float()

        print(f"Loading V_pretrain from {V_pretrain_path}...")
        V.load_state_dict(torch.load(V_pretrain_path, map_location=device, weights_only=True))
        print("V dtype:", next(V.parameters()).dtype)

        result = finetune_value_w_MILP(
            V_model=V,
            f_pi_model=f_pi,
            h_model=h,
            x_low=CartPole.x_low.astype(np.float32),
            x_high=CartPole.x_high.astype(np.float32),
            lr=finetune_config["lr"],
            max_iter=finetune_config["max_iter"],
            search_size=finetune_config["search_size"],
            bnd_ratio=finetune_config["bnd_ratio"],
            bnd_ratio_avg=finetune_config["bnd_ratio_avg"],
            min_bnd_ratio=finetune_config["min_bnd_ratio"],
            max_bnd_ratio=finetune_config["max_bnd_ratio"],
            bnd_eps=finetune_config["bnd_eps"],
            search_method=finetune_config["search_method"],
            pgd_step=finetune_config["pgd_step"],
            pgd_eps=finetune_config["pgd_eps"],
            backtrack_step=finetune_config["backtrack_step"],
            length_discount=finetune_config["length_discount"],
            direct_discount=finetune_config["direct_discount"],
            tol=finetune_config["tol"],
            capacity=finetune_config["capacity"],
            sample_size=finetune_config["sample_size"],
            search_stop=finetune_config["search_stop"],
            replay=finetune_config["replay"],
            max_skip=finetune_config["max_skip"],
            reg_method=finetune_config["reg_method"],
            eps_h=finetune_config["eps_h"],
            eps_v=finetune_config["eps_v"],
            reg_coef=finetune_config["reg_coef"],
            milp_timeout=finetune_config["milp_timeout"],
            milp_start_iter=finetune_config["milp_start_iter"],
            milp_box_shrink=finetune_config["milp_box_shrink"],
            milp_box_grow=finetune_config["milp_box_grow"],
            milp_box_max=finetune_config["milp_box_max"],
            milp_max_inconclusive=finetune_config["milp_max_inconclusive"],
            log_dir=log_dir,
            eval_every=finetune_config["eval_every"],
            save_every=finetune_config["save_every"],
            save_path=run_final_model_path,
            device=device,
            seed=seed,
        )

        # Backward-compatible handling:
        # old version -> (log_path, tfr, iterations)
        # new version -> (log_path, tfr, iterations, stats)
        if isinstance(result, tuple) and len(result) == 4:
            log_path, tfr, iterations, milp_stats = result
        elif isinstance(result, tuple) and len(result) == 3:
            log_path, tfr, iterations = result
            milp_stats = {
                "note": "finetune_value_w_MILP does not yet return MILP stats. Update the function to expose them."
            }
        else:
            raise RuntimeError(f"Unexpected return from finetune_value_w_MILP: {type(result)} / {result}")

        print("\nReturn from finetune_value_w_MILP:")
        print("log_path:", log_path)
        print("tfr:", tfr)
        print("iterations:", iterations)
        print("milp_stats:", milp_stats)

        tfrs.append(float(tfr))
        iters.append(int(iterations))

        atomic_torch_save(V.state_dict(), run_snapshot_path)
        print(f"Saved finetuned V snapshot to: {run_snapshot_path}")

        run_summary = {
            "system": "cart_pole",
            "seed": seed,
            "timestamp": run_timestamp,
            "device": device,
            "data_path": os.path.abspath(data_path),
            "f_path": os.path.abspath(f_path),
            "h_path": os.path.abspath(h_path),
            "V_pretrain_path": os.path.abspath(V_pretrain_path),
            "pretrain_snapshot_path": os.path.abspath(pretrain_snapshot_path),
            "final_model_path": os.path.abspath(run_final_model_path),
            "timestamped_model_path": os.path.abspath(run_snapshot_path),
            "log_path": os.path.abspath(log_path),
            "tfr": float(tfr),
            "iterations": int(iterations),
            "used_saved_controller": True,
            "policy_source": "CartPole.create_policy()",
            "milp_stats": milp_stats,
            "finetune_config": {
                **finetune_config,
                "seed": seed,
            },
        }

        save_json(run_summary, run_summary_path)
        print(f"Saved run summary to: {run_summary_path}")

        run_summaries.append(run_summary)

    # -----------------------------
    # Aggregate summary
    # -----------------------------
    aggregate_summary = {
        "system": "cart_pole",
        "timestamp": timestamp,
        "device": device,
        "seeds": seeds,
        "data_path": os.path.abspath(data_path),
        "f_path": os.path.abspath(f_path),
        "h_path": os.path.abspath(h_path),
        "V_pretrain_path": os.path.abspath(V_pretrain_path),
        "pretrain_snapshot_path": os.path.abspath(pretrain_snapshot_path),
        "mean_tfr": float(np.mean(tfrs)),
        "std_tfr": float(np.std(tfrs)),
        "mean_tfr_percent": float(np.mean(tfrs) * 100.0),
        "std_tfr_percent": float(np.std(tfrs) * 100.0),
        "mean_iterations": float(np.mean(iters)),
        "std_iterations": float(np.std(iters)),
        "finetune_config": finetune_config,
        "runs": run_summaries,
    }

    save_json(aggregate_summary, aggregate_summary_path)
    print(f"\nSaved aggregate summary to: {aggregate_summary_path}")

    print("\nTraining complete.")
    print(f"Mean TFR: {np.mean(tfrs) * 100:.4f}% (std: {np.std(tfrs) * 100:.4f})")
    print(f"Mean iterations: {np.mean(iters):.2f} (std: {np.std(iters):.2f})")


if __name__ == "__main__":
    main()