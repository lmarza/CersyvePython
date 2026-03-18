"""
Finetune V for CartPole (Enumeration + MILP) + FORCED SAVE of all .pth artifacts.

Artifacts produced:
- models/cart_pole/f_snapshot.pth
- models/cart_pole/h_snapshot.pth
- models/cart_pole/V_pretrain_snapshot.pth
- models/cart_pole/V_finetune_enum_milp.pth   (final)
- models/cart_pole/V_finetune_enum_milp_FORCED.pth
- logs/cart_pole/finetune_enum_milp_YYYYmmdd_HHMMSS/   (tensorboard + checkpoints)
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
    """
    Save to a temporary file, then rename -> reduces risk of corrupted checkpoints.
    """
    ensure_dir(os.path.dirname(path))
    tmp_path = path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX


def force_model_float32(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """
    Force model parameters and buffers to float32 and move to device.
    """
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
    """
    Return dtype of first parameter, otherwise first buffer, otherwise float32.
    """
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
    """
    Force-save the main artifacts into models_dir with absolute paths printed.
    Returns dict of saved paths.
    """
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
    ensure_dir(os.path.dirname(meta_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    out["meta"] = meta_path

    print("\nFORCED SAVE completed:")
    for k, p in out.items():
        print(f"  {k}: {os.path.abspath(p)}")

    return out


def main():
    # -----------------------------
    # Paths
    # -----------------------------
    data_path = "../../data/cart_pole_data.npz"
    models_dir = "../../models/cart_pole/"
    logs_dir = "../../logs/cart_pole/"
    ensure_dir(models_dir)
    ensure_dir(logs_dir)

    # -----------------------------
    # Device
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------------
    # Determinism / seeds
    # -----------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    # -----------------------------
    # Force state bounds to float32
    # -----------------------------
    x_low = np.asarray(CartPole.x_low, dtype=np.float32)
    x_high = np.asarray(CartPole.x_high, dtype=np.float32)

    # -----------------------------
    # Load dataset stats (normalization)
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
    # Load dynamics model f
    # -----------------------------
    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_path = os.path.join(models_dir, "f.pth")
    print(f"Loading dynamics model from {f_path}...")
    f_model.load_state_dict(torch.load(f_path, map_location=device))
    f_model = force_model_float32(f_model, device)
    f_model.eval()

    # -----------------------------
    # Load policy model pi
    # -----------------------------
    pi_model = CartPole.create_policy()
    pi_model = force_model_float32(pi_model, device)
    pi_model.eval()

    # -----------------------------
    # Closed-loop dynamics f_pi
    # -----------------------------
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

    # -----------------------------
    # Load constraint model h
    # -----------------------------
    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_path = os.path.join(models_dir, "h.pth")
    print(f"Loading constraint model from {h_path}...")
    h_model.load_state_dict(torch.load(h_path, map_location=device))
    h_model = force_model_float32(h_model, device)
    h_model.eval()

    # -----------------------------
    # Load pretrained value model V
    # -----------------------------
    V_model = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_pre_path = os.path.join(models_dir, "V_pretrain.pth")
    print(f"Loading pretrained value function from {V_pre_path}...")
    V_model.load_state_dict(torch.load(V_pre_path, map_location=device))
    V_model = force_model_float32(V_model, device)
    V_model.train()  # finetune updates params

    # -----------------------------
    # Dtype diagnostics
    # -----------------------------
    print("\nDtype diagnostics:")
    print("  pi dtype   :", get_model_dtype(pi_model))
    print("  f dtype    :", get_model_dtype(f_model))
    print("  f_pi dtype :", get_model_dtype(f_pi_model))
    print("  h dtype    :", get_model_dtype(h_model))
    print("  V dtype    :", get_model_dtype(V_model))
    print("  x_low dtype :", x_low.dtype)
    print("  x_high dtype:", x_high.dtype)

    # -----------------------------
    # Snapshot pretrained V
    # -----------------------------
    V_pre_snapshot = os.path.join(models_dir, "V_pretrain_snapshot.pth")
    atomic_torch_save(V_model.state_dict(), V_pre_snapshot)
    print("Snapshot saved:", os.path.abspath(V_pre_snapshot))

    # -----------------------------
    # Run finetuning (Enum + MILP)
    # -----------------------------
    save_path = os.path.join(models_dir, "V_finetune_enum_milp.pth")

    finetune_cfg = dict(
        lr=1e-4,
        max_iter=200000,
        tol=1e-4,

        capacity=20000,
        sample_size=256,
        replay=1,
        max_skip=500,

        # Enumeration
        enum_freq=2000,
        enum_n_samples=30000,
        enum_coverage=70.0,
        enum_n_trees=800,
        enum_max_depth=14,
        enum_n_train=60000,
        refill_threshold=1000,

        # Fallback (quando buffer vuoto)
        fallback_eval_rounds=3,
        fallback_eval_samples=60000,
        fallback_n_neighbors=8000,
        fallback_radius=0.05,

        # Regularization (ESR)
        search_size=4000,
        reg_method="ESR",
        eps_h=0.01,
        eps_v=0.01,
        reg_coef=0.1,

        # MILP
        milp_timeout=300,

        # Logging
        log_dir=logs_dir,
        eval_every=100,
        save_every=5000,
        save_path=save_path,
        device=device,
        seed=42,
    )

    print("\nFinetuning value function (enum+MILP)...")
    log_path, tfr, iterations = finetune_value_enum_MILP(
        V_model=V_model,
        f_pi_model=f_pi_model,
        h_model=h_model,
        x_low=x_low,
        x_high=x_high,
        **finetune_cfg
    )

    print("\nReturn from finetune_value_enum_MILP:")
    print("log_path:", log_path)
    print("tfr:", float(tfr))
    print("iterations:", int(iterations))
    print("Expected final V saved at:", os.path.abspath(save_path))

    # -----------------------------
    # FORCED SAVE (final safety net)
    # -----------------------------
    extra_meta = {
        "log_path": os.path.abspath(log_path),
        "save_path": os.path.abspath(save_path),
        "tfr": float(tfr),
        "iterations": int(iterations),
        "finetune_cfg": finetune_cfg,
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

    force_save_artifacts(
        models_dir=models_dir,
        f_model=f_model,
        h_model=h_model,
        V_model=V_model,
        V_save_name="V_finetune_enum_milp_FORCED.pth",
        extra_meta=extra_meta,
    )

    print("\nDone.")
    print(f"TensorBoard:\n  tensorboard --logdir={os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()