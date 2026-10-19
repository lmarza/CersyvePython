"""
Train dynamics model for the RobotArm system.

RobotArm follows the benchmark formulation:
- linear dynamics
- nonlinear constraint

Even though the analytical dynamics are known exactly, we still train a neural
dynamics model here to keep the same approximation pipeline used for the other systems.
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

from systems.robot_arm import RobotArm
from cersyve import create_mlp, train_dynamics


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


# -----------------------------
# Load data
# -----------------------------
data_path = "../../data/robot_arm_data.npz"
print(f"Loading data from {data_path}...")
data_npz = np.load(data_path)
data = {key: data_npz[key] for key in data_npz.keys()}

print("\nLoaded dataset keys:")
for k, v in data.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")


# -----------------------------
# Create dynamics model
# -----------------------------
dynamics_hidden_sizes = [32, 32]

f_model = create_mlp(
    RobotArm.x_dim + RobotArm.u_dim,
    RobotArm.x_dim,
    dynamics_hidden_sizes
)

print("\nDynamics model architecture:")
print(f"  Input dim: {RobotArm.x_dim + RobotArm.u_dim}")
print(f"  Hidden sizes: {dynamics_hidden_sizes}")
print(f"  Output dim: {RobotArm.x_dim}  (predicting dx)")
print(f"  Total parameters: {sum(p.numel() for p in f_model.parameters())}")


# -----------------------------
# Train dynamics model
# -----------------------------
print("\nTraining dynamics model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# State-action space size for APA noise scaling
xu_space_size = np.concatenate([
    RobotArm.x_high - RobotArm.x_low,
    RobotArm.u_high - RobotArm.u_low
]).astype(np.float32)

log_path = train_dynamics(
    data=data,
    f_model=f_model,
    closed_loop=False,
    lr=1e-3,
    batch_size=256,
    epoch_num=100,
    weight_decay=0.0,
    penalty="APA",
    noise_scale=0.1,
    space_size=xu_space_size,
    apa_coef=1e-4,
    log_dir="../../logs/robot_arm/",
    device=device,
    seed=seed,
    task="robot_arm"
)


# -----------------------------
# Final save (robust)
# -----------------------------
models_dir = "../../models/robot_arm/"
ensure_dir(models_dir)

f_path = os.path.join(models_dir, "f.pth")
atomic_torch_save(f_model.state_dict(), f_path)


# -----------------------------
# Quick post-training sanity check
# -----------------------------
print("\nRunning quick post-training sanity check...")

# We try to infer common key names used in the dataset
possible_x_keys = ["x", "X", "state", "states"]
possible_u_keys = ["u", "U", "action", "actions"]
possible_dx_keys = ["dx", "DX", "delta", "dynamics_target"]
possible_xnext_keys = ["x_next", "X_next", "next_state", "next_states", "y", "Y"]

x_key = next((k for k in possible_x_keys if k in data), None)
u_key = next((k for k in possible_u_keys if k in data), None)
dx_key = next((k for k in possible_dx_keys if k in data), None)
xnext_key = next((k for k in possible_xnext_keys if k in data), None)

sanity_metrics = {}

if x_key is not None and u_key is not None and (dx_key is not None or xnext_key is not None):
    X = data[x_key]
    U = data[u_key]

    # Normalize shape to [dim, N]
    if X.shape[0] != RobotArm.x_dim and X.shape[1] == RobotArm.x_dim:
        X = X.T
    if U.shape[0] != RobotArm.u_dim and U.shape[1] == RobotArm.u_dim:
        U = U.T

    if dx_key is not None:
        DX = data[dx_key]
        if DX.shape[0] != RobotArm.x_dim and DX.shape[1] == RobotArm.x_dim:
            DX = DX.T
        target_dx = DX
    else:
        X_next = data[xnext_key]
        if X_next.shape[0] != RobotArm.x_dim and X_next.shape[1] == RobotArm.x_dim:
            X_next = X_next.T
        target_dx = (X_next - X).astype(np.float32)

    n_eval = min(5000, X.shape[1])
    idx = np.random.choice(X.shape[1], size=n_eval, replace=False)

    X_eval = X[:, idx].astype(np.float32)
    U_eval = U[:, idx].astype(np.float32)
    DX_eval = target_dx[:, idx].astype(np.float32)

    xu_eval = np.concatenate([X_eval, U_eval], axis=0).T
    xu_eval_t = torch.from_numpy(xu_eval).to(device)

    f_model = f_model.to(device).eval()
    with torch.no_grad():
        DX_pred = f_model(xu_eval_t).cpu().numpy().T

    abs_err = np.abs(DX_pred - DX_eval)
    mse = float(np.mean((DX_pred - DX_eval) ** 2))
    mae = float(np.mean(abs_err))
    max_err = float(np.max(abs_err))

    sanity_metrics = {
        "n_eval": int(n_eval),
        "mse_dx": mse,
        "mae_dx": mae,
        "max_abs_err_dx": max_err,
        "x_key": x_key,
        "u_key": u_key,
        "target_source": dx_key if dx_key is not None else xnext_key,
    }

    print("\nSanity check on dynamics model:")
    print(f"  target source: {sanity_metrics['target_source']}")
    print(f"  n_eval:        {n_eval}")
    print(f"  mse(dx):       {mse:.8f}")
    print(f"  mae(dx):       {mae:.8f}")
    print(f"  max|err|(dx):  {max_err:.8f}")
else:
    print("Could not run sanity check automatically: dataset keys not recognized.")


# -----------------------------
# Save summary
# -----------------------------
summary_path = os.path.join(models_dir, "f_train_summary.json")
summary = {
    "system": "robot_arm",
    "device": device,
    "data_path": os.path.abspath(data_path),
    "model_path": os.path.abspath(f_path),
    "hidden_sizes": dynamics_hidden_sizes,
    "seed": seed,
    "train_config": {
        "lr": 1e-3,
        "batch_size": 256,
        "epoch_num": 100,
        "weight_decay": 0.0,
        "penalty": "APA",
        "noise_scale": 0.1,
        "apa_coef": 1e-4,
        "closed_loop": False,
    },
    "sanity_metrics": sanity_metrics,
    "log_path": os.path.abspath(log_path),
}
save_json(summary, summary_path)


print("\nTraining complete!")
print("Model saved to:", os.path.abspath(f_path))
print("Summary saved to:", os.path.abspath(summary_path))
print("Logs saved to:", os.path.abspath(log_path))
print("\nTo view training curves, run:")
print(f"  tensorboard --logdir={os.path.abspath(log_path)}")
