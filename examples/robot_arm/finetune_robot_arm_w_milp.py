import os
import sys
import json
from datetime import datetime

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from systems.robot_arm import RobotArm
from cersyve import create_mlp
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

    model_dir = os.path.join(base_dir, "models", "robot_arm")
    log_dir = os.path.join(base_dir, "logs", "robot_arm")
    ensure_dir(model_dir)
    ensure_dir(log_dir)

    h_path = os.path.join(model_dir, "h.pth")
    V_pretrain_path = os.path.join(model_dir, "V_pretrain.pth")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate_summary_path = os.path.join(
        model_dir,
        f"V_finetuned_milp_aggregate_{timestamp}.json"
    )

    # -----------------------------
    # Build benchmark-aligned models
    # -----------------------------
    pi = RobotArm.create_policy().to(device).float().eval()
    f_pi = RobotArm.create_closed_loop_dynamics_model().to(device).float().eval()

    constraint_hidden_sizes = [16, 16]
    value_hidden_sizes = [32, 32]

    h = create_mlp(
        RobotArm.x_dim,
        1,
        constraint_hidden_sizes
    ).to(device).float()

    print(f"Loading h from {h_path}...")
    h.load_state_dict(torch.load(h_path, map_location=device, weights_only=True))
    h = h.eval()

    print("pi dtype:", next(pi.parameters()).dtype)
    print("h dtype:", next(h.parameters()).dtype)

    # -----------------------------
    # Finetuning config
    # -----------------------------
    # Kept close to your current successful setup, while remaining
    # consistent with the benchmark formulation.
    finetune_config = {
        "lr": 1e-4,
        "max_iter": 15000,
        "search_size": 1000,
        "sample_size": 1000,
        "capacity": 10000,
        "replay": 1,

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
        "search_stop": 100,

        "reg_method": "ESR",
        "eps_h": 0.01,
        "eps_v": 0.01,
        "reg_coef": 0.1,

        "milp_timeout": 60,
        "milp_start_iter": 8000,
        "max_skip": 200,
        "eval_every": 10,
        "save_every": 1000,

        # keep MILP directly on the full box.
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
        run_snapshot_path = os.path.join(
            model_dir,
            f"V_finetuned_milp_seed{seed}_{run_timestamp}.pth"
        )
        run_summary_path = os.path.join(
            model_dir,
            f"V_finetuned_milp_seed{seed}_{run_timestamp}.json"
        )

        # Rebuild V from the same pretrained checkpoint for each seed
        V = create_mlp(
            RobotArm.x_dim,
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
            x_low=RobotArm.x_low.astype(np.float32),
            x_high=RobotArm.x_high.astype(np.float32),
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

        # Backward-compatible handling
        if isinstance(result, tuple) and len(result) == 4:
            log_path, tfr, iterations, milp_stats = result
        elif isinstance(result, tuple) and len(result) == 3:
            log_path, tfr, iterations = result
            milp_stats = {
                "note": "finetune_value_w_MILP does not yet return MILP stats. "
                        "Update the function if you want explicit MILP statistics."
            }
        else:
            raise RuntimeError(
                f"Unexpected return from finetune_value_w_MILP: {type(result)} / {result}"
            )

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
            "system": "robot_arm",
            "seed": seed,
            "timestamp": run_timestamp,
            "device": device,
            "h_path": os.path.abspath(h_path),
            "V_pretrain_path": os.path.abspath(V_pretrain_path),
            "final_model_path": os.path.abspath(run_final_model_path),
            "timestamped_model_path": os.path.abspath(run_snapshot_path),
            "log_path": os.path.abspath(log_path),
            "tfr": float(tfr),
            "iterations": int(iterations),
            "used_saved_controller": True,
            "used_learned_dynamics_model": False,
            "policy_source": "RobotArm.create_policy()",
            "dynamics_source": "RobotArm.create_closed_loop_dynamics_model()",
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
        "system": "robot_arm",
        "timestamp": timestamp,
        "device": device,
        "seeds": seeds,
        "h_path": os.path.abspath(h_path),
        "V_pretrain_path": os.path.abspath(V_pretrain_path),
        "dynamics_source": "RobotArm.create_closed_loop_dynamics_model()",
        "used_learned_dynamics_model": False,
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