"""
Finetune V for CartPole (Enumeration + MILP) — multi-seed + forced save + JSON summaries.

Artifacts produced:
- models/cart_pole/f_snapshot.pth
- models/cart_pole/h_snapshot.pth
- models/cart_pole/V_pretrain_snapshot.pth
- models/cart_pole/V_finetune_enum_milp_seed{seed}.pth
- models/cart_pole/V_finetuned_enum_milp_seed{seed}_{timestamp}.pth
- models/cart_pole/V_finetuned_enum_milp_seed{seed}_{timestamp}.json
- models/cart_pole/V_finetuned_enum_milp_aggregate_{timestamp}.json
- logs/cart_pole/finetune_enum_milp_YYYYmmdd_HHMMSS/
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
from cersyve.finetune import finetune_value_enum_MILP


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


def force_model_float32(model: torch.nn.Module, device: str) -> torch.nn.Module:
    model = model.to(device)
    model = model.float()

    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

    for b in model.buffers():
        b.data = b.data.float()

    return model


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    for p in model.parameters():
        return p.dtype
    for b in model.buffers():
        return b.dtype
    return torch.float32


def force_save_artifacts(
    models_dir: str,
    f_model: torch.nn.Module,
    h_model: torch.nn.Module,
    V_model: torch.nn.Module,
    V_save_name: str,
    extra_meta: dict | None = None,
) -> dict:
    ensure_dir(models_dir)

    out = {}
    out["f_snapshot"] = os.path.join(models_dir, "f_snapshot.pth")
    out["h_snapshot"] = os.path.join(models_dir, "h_snapshot.pth")
    out["V_final"] = os.path.join(models_dir, V_save_name)

    atomic_torch_save(f_model.state_dict(), out["f_snapshot"])
    atomic_torch_save(h_model.state_dict(), out["h_snapshot"])
    atomic_torch_save(V_model.state_dict(), out["V_final"])

    meta = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "cartpole": {
            "x_dim": int(CartPole.x_dim),
            "u_dim": int(CartPole.u_dim),
            "x_low": np.asarray(CartPole.x_low).tolist(),
            "x_high": np.asarray(CartPole.x_high).tolist(),
        },
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = os.path.join(models_dir, "run_meta.json")
    save_json(meta, meta_path)
    out["meta"] = meta_path

    print("\nFORCED SAVE completed:")
    for k, p in out.items():
        print(f"  {k}: {os.path.abspath(p)}")

    return out


def main():
    # -----------------------------
    # Paths
    # -----------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(base_dir, "data", "cart_pole_data.npz")
    models_dir = os.path.join(base_dir, "models", "cart_pole")
    logs_dir = os.path.join(base_dir, "logs", "cart_pole")
    ensure_dir(models_dir)
    ensure_dir(logs_dir)

    # -----------------------------
    # Device
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------------
    # Determinism base
    # -----------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    # -----------------------------
    # State bounds float32
    # -----------------------------
    x_low = np.asarray(CartPole.x_low, dtype=np.float32)
    x_high = np.asarray(CartPole.x_high, dtype=np.float32)

    # -----------------------------
    # Load dataset stats
    # -----------------------------
    print(f"Loading data from {data_path}...")
    data_npz = np.load(data_path)
    data = {}
    for k in data_npz.keys():
        arr = data_npz[k]
        if np.issubdtype(arr.dtype, np.floating):
            data[k] = arr.astype(np.float32)
        else:
            data[k] = arr

    # -----------------------------
    # Build/load shared models
    # -----------------------------
    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_path = os.path.join(models_dir, "f.pth")
    print(f"Loading dynamics model from {f_path}...")
    f_model.load_state_dict(torch.load(f_path, map_location=device))
    f_model = force_model_float32(f_model, device)
    f_model.eval()

    pi_model = CartPole.create_policy()
    pi_model = force_model_float32(pi_model, device)
    pi_model.eval()

    f_pi_model = create_closed_loop_dynamics_model(
        f_model,
        pi_model,
        data,
        x_low,
        x_high,
        CartPole.u_dim
    )
    f_pi_model = force_model_float32(f_pi_model, device)
    f_pi_model.eval()
    print("Created closed-loop dynamics model")

    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_path = os.path.join(models_dir, "h.pth")
    print(f"Loading constraint model from {h_path}...")
    h_model.load_state_dict(torch.load(h_path, map_location=device))
    h_model = force_model_float32(h_model, device)
    h_model.eval()

    # -----------------------------
    # Dtype diagnostics
    # -----------------------------
    print("\nDtype diagnostics:")
    print("  pi dtype   :", get_model_dtype(pi_model))
    print("  f dtype    :", get_model_dtype(f_model))
    print("  f_pi dtype :", get_model_dtype(f_pi_model))
    print("  h dtype    :", get_model_dtype(h_model))
    print("  x_low dtype :", x_low.dtype)
    print("  x_high dtype:", x_high.dtype)

    # -----------------------------
    # Snapshot pretrained V
    # -----------------------------
    V_pre_path = os.path.join(models_dir, "V_pretrain.pth")
    V_tmp = create_mlp(CartPole.x_dim, 1, [64, 64])
    print(f"Loading pretrained value function from {V_pre_path}...")
    V_tmp.load_state_dict(torch.load(V_pre_path, map_location=device))
    V_tmp = force_model_float32(V_tmp, device)

    V_pre_snapshot = os.path.join(models_dir, "V_pretrain_snapshot.pth")
    atomic_torch_save(V_tmp.state_dict(), V_pre_snapshot)
    print("Snapshot saved:", os.path.abspath(V_pre_snapshot))

    # -----------------------------
    # Config
    # -----------------------------
    # con seed = [42] si va 5 min e 67%tfr
    #metto i parametri di questo modello così nei commenti per non perdere le modifiche
    finetune_cfg = dict(
        lr=1e-4,
        max_iter=5000,#5000
        tol=1e-4,

        capacity=10000,
        sample_size=100,
        replay=1,
        max_skip=200,#1000 ##500

        # Enumeration
        enum_freq=1000,
        enum_n_samples=10000, ##20000
        enum_coverage=70.0,
        enum_n_trees=500,#200
        enum_max_depth=20,
        enum_n_train=30000,#40000
        refill_threshold=200, ##500

        # Fallback
        fallback_eval_rounds=2,#5 ##3
        fallback_eval_samples=30000,#100000 ##50000
        fallback_n_neighbors=8000,
        fallback_radius=0.05,

        # Regularization
        search_size=2000,
        reg_method="ESR",
        eps_h=0.01,
        eps_v=0.01,
        reg_coef=0.1,

        # MILP
        milp_timeout=120,# 60

        # Logging
        log_dir=logs_dir,
        eval_every=100,
        save_every=5000,
        device=device
    )

    # -----------------------------
    # Multi-seed runs
    # -----------------------------
    seeds = [1, 0, 2]
    tfrs = []
    iters = []
    run_summaries = []

    aggregate_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate_summary_path = os.path.join(
        models_dir,
        f"V_finetuned_enum_milp_aggregate_{aggregate_timestamp}.json"
    )

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"RUNNING SEED {seed}")
        print("=" * 70)

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_final_model_path = os.path.join(models_dir, f"V_finetune_enum_milp_seed{seed}.pth")
        run_snapshot_path = os.path.join(
            models_dir, f"V_finetuned_enum_milp_seed{seed}_{run_timestamp}.pth"
        )
        run_summary_path = os.path.join(
            models_dir, f"V_finetuned_enum_milp_seed{seed}_{run_timestamp}.json"
        )

        # Rebuild V from pretrained checkpoint for each seed
        V_model = create_mlp(CartPole.x_dim, 1, [64, 64])
        print(f"Loading pretrained value function from {V_pre_path}...")
        V_model.load_state_dict(torch.load(V_pre_path, map_location=device))
        V_model = force_model_float32(V_model, device)
        V_model.train()

        print("V dtype:", get_model_dtype(V_model))

        result = finetune_value_enum_MILP(
            V_model=V_model,
            f_pi_model=f_pi_model,
            h_model=h_model,
            x_low=x_low,
            x_high=x_high,
            save_path=run_final_model_path,
            seed=seed,
            **finetune_cfg,
        )

        print("\nReturn from finetune_value_enum_MILP:")
        print("result:", result)

        # Backward-compatible handling
        if isinstance(result, tuple) and len(result) == 4:
            log_path, tfr, iterations, enum_stats = result
        elif isinstance(result, tuple) and len(result) == 3:
            log_path, tfr, iterations = result
            enum_stats = {
                "note": "finetune_value_enum_MILP does not yet return extra stats."
            }
        else:
            raise RuntimeError(f"Unexpected return from finetune_value_enum_MILP: {type(result)} / {result}")

        print("\nParsed return from finetune_value_enum_MILP:")
        print("log_path:", log_path)
        print("tfr:", tfr)
        print("iterations:", iterations)
        print("enum_stats:", enum_stats)

        tfrs.append(float(tfr))
        iters.append(int(iterations))

        atomic_torch_save(V_model.state_dict(), run_snapshot_path)
        print(f"Saved finetuned V snapshot to: {run_snapshot_path}")

        run_summary = {
            "system": "cart_pole",
            "method": "enum_milp",
            "seed": seed,
            "timestamp": run_timestamp,
            "device": device,
            "data_path": os.path.abspath(data_path),
            "f_path": os.path.abspath(f_path),
            "h_path": os.path.abspath(h_path),
            "V_pretrain_path": os.path.abspath(V_pre_path),
            "pretrain_snapshot_path": os.path.abspath(V_pre_snapshot),
            "final_model_path": os.path.abspath(run_final_model_path),
            "timestamped_model_path": os.path.abspath(run_snapshot_path),
            "log_path": os.path.abspath(log_path),
            "tfr": float(tfr),
            "iterations": int(iterations),
            "policy_source": "CartPole.create_policy()",
            "enum_stats": enum_stats,
            "finetune_config": {
                **finetune_cfg,
                "save_path": os.path.abspath(run_final_model_path),
                "seed": seed,
            },
            "dtype_info": {
                "pi_model": str(get_model_dtype(pi_model)),
                "f_model": str(get_model_dtype(f_model)),
                "f_pi_model": str(get_model_dtype(f_pi_model)),
                "h_model": str(get_model_dtype(h_model)),
                "V_model": str(get_model_dtype(V_model)),
                "x_low": str(x_low.dtype),
                "x_high": str(x_high.dtype),
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
        "method": "enum_milp",
        "timestamp": aggregate_timestamp,
        "device": device,
        "seeds": seeds,
        "data_path": os.path.abspath(data_path),
        "f_path": os.path.abspath(f_path),
        "h_path": os.path.abspath(h_path),
        "V_pretrain_path": os.path.abspath(V_pre_path),
        "pretrain_snapshot_path": os.path.abspath(V_pre_snapshot),
        "mean_tfr": float(np.mean(tfrs)),
        "std_tfr": float(np.std(tfrs)),
        "mean_tfr_percent": float(np.mean(tfrs) * 100.0),
        "std_tfr_percent": float(np.std(tfrs) * 100.0),
        "mean_iterations": float(np.mean(iters)),
        "std_iterations": float(np.std(iters)),
        "finetune_config": finetune_cfg,
        "runs": run_summaries,
    }

    save_json(aggregate_summary, aggregate_summary_path)
    print(f"\nSaved aggregate summary to: {aggregate_summary_path}")

    # -----------------------------
    # Forced save safety net
    # -----------------------------
    extra_meta = {
        "aggregate_summary_path": os.path.abspath(aggregate_summary_path),
        "mean_tfr": float(np.mean(tfrs)),
        "std_tfr": float(np.std(tfrs)),
        "mean_iterations": float(np.mean(iters)),
        "std_iterations": float(np.std(iters)),
        "seeds": seeds,
        "finetune_cfg": finetune_cfg,
        "dtype_info": {
            "pi_model": str(get_model_dtype(pi_model)),
            "f_model": str(get_model_dtype(f_model)),
            "f_pi_model": str(get_model_dtype(f_pi_model)),
            "h_model": str(get_model_dtype(h_model)),
            "x_low": str(x_low.dtype),
            "x_high": str(x_high.dtype),
        },
    }

    # Save last shared models + last V for safety
    force_save_artifacts(
        models_dir=models_dir,
        f_model=f_model,
        h_model=h_model,
        V_model=V_model,
        V_save_name="V_finetune_enum_milp_FORCED.pth",
        extra_meta=extra_meta,
    )

    print("\nTraining complete.")
    print(f"Mean True Feasibility Region (TFR): {np.mean(tfrs)*100:.4f}% (std: {np.std(tfrs)*100:.4f})")
    print(f"Mean iterations: {np.mean(iters):.2f} (std: {np.std(iters):.2f})")


if __name__ == "__main__":
    main()