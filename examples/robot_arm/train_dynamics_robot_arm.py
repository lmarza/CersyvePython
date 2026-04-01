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
# Following the other benchmark scripts:
# two hidden layers of 32 neurons.
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





# """
# Train dynamics model for the RobotArm system.
#
# RobotArm follows the benchmark formulation:
# - linear dynamics
# - nonlinear constraint
#
# Even though the analytical dynamics are known exactly, we still train a neural
# dynamics model here to keep the same approximation pipeline used for the other systems.
# """
#
# import os
# import sys
# import json
#
# # Add project root to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
#
# import numpy as np
# import torch
#
# from systems.robot_arm import RobotArm
# from cersyve import create_mlp, train_dynamics
#
#
# # -----------------------------
# # Robust saving utilities
# # -----------------------------
# def ensure_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)
#
#
# def atomic_torch_save(state_dict, path: str) -> None:
#     ensure_dir(os.path.dirname(path))
#     tmp_path = path + ".tmp"
#     torch.save(state_dict, tmp_path)
#     os.replace(tmp_path, path)
#
#
# def save_json(obj: dict, path: str) -> None:
#     ensure_dir(os.path.dirname(path))
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2)
#
#
# # -----------------------------
# # Detailed sanity check
# # -----------------------------
# @torch.no_grad()
# def sanity_check_dynamics_detailed(
#     f_model,
#     dataset,
#     device="cpu",
#     n_eval=5000,
#     target_source="dx",   # "dx" oppure "x_next"
#     top_k=10,
# ):
#     """
#     Detailed sanity check for a learned dynamics model.
#
#     Assumptions:
#     - dataset["x"]      : current states, shape [N, state_dim]
#     - dataset["u"]      : actions,        shape [N, action_dim]
#     - dataset["x_next"] : next states,    shape [N, state_dim]   (optional if dx is present)
#     - dataset["dx"]     : state delta,    shape [N, state_dim]   (optional if x_next is present)
#
#     If target_source == "dx", the target is dataset["dx"] if present,
#     otherwise (x_next - x).
#     If target_source == "x_next", the target is x_next directly.
#     """
#
#     assert target_source in {"dx", "x_next"}, "target_source must be 'dx' or 'x_next'"
#
#     x_all = dataset["x"]
#     u_all = dataset["u"]
#
#     if isinstance(x_all, np.ndarray):
#         x_all = torch.from_numpy(x_all).float()
#     if isinstance(u_all, np.ndarray):
#         u_all = torch.from_numpy(u_all).float()
#
#     if "dx" in dataset and dataset["dx"] is not None:
#         dx_all = dataset["dx"]
#         if isinstance(dx_all, np.ndarray):
#             dx_all = torch.from_numpy(dx_all).float()
#     else:
#         dx_all = None
#
#     if "x_next" in dataset and dataset["x_next"] is not None:
#         x_next_all = dataset["x_next"]
#         if isinstance(x_next_all, np.ndarray):
#             x_next_all = torch.from_numpy(x_next_all).float()
#     else:
#         x_next_all = None
#
#     n_total = x_all.shape[0]
#     n_eval = min(n_eval, n_total)
#
#     idx = torch.randperm(n_total)[:n_eval]
#     x = x_all[idx].to(device)
#     u = u_all[idx].to(device)
#
#     if target_source == "dx":
#         if dx_all is not None:
#             y_true = dx_all[idx].to(device)
#         else:
#             assert x_next_all is not None, "Need either dx or x_next in dataset"
#             x_next = x_next_all[idx].to(device)
#             y_true = x_next - x
#     else:
#         assert x_next_all is not None, "target_source='x_next' requires x_next in dataset"
#         y_true = x_next_all[idx].to(device)
#
#     xu = torch.cat([x, u], dim=1)
#     y_pred = f_model(xu)
#
#     err = y_pred - y_true
#     abs_err = err.abs()
#
#     # Global metrics
#     mse = (err ** 2).mean().item()
#     mae = abs_err.mean().item()
#     max_err = abs_err.max().item()
#     p95 = torch.quantile(abs_err.flatten(), 0.95).item()
#     p99 = torch.quantile(abs_err.flatten(), 0.99).item()
#
#     print("\nSanity check on dynamics model:")
#     print(f"  target source: {target_source}")
#     print(f"  n_eval:        {n_eval}")
#     print(f"  mse({target_source}):       {mse:.8f}")
#     print(f"  mae({target_source}):       {mae:.8f}")
#     print(f"  p95|err|({target_source}):  {p95:.8f}")
#     print(f"  p99|err|({target_source}):  {p99:.8f}")
#     print(f"  max|err|({target_source}):  {max_err:.8f}")
#
#     # Per-dimension metrics
#     per_dim_mse = (err ** 2).mean(dim=0).detach().cpu().numpy()
#     per_dim_mae = abs_err.mean(dim=0).detach().cpu().numpy()
#     per_dim_p95 = torch.quantile(abs_err, 0.95, dim=0).detach().cpu().numpy()
#     per_dim_p99 = torch.quantile(abs_err, 0.99, dim=0).detach().cpu().numpy()
#     per_dim_max = abs_err.max(dim=0).values.detach().cpu().numpy()
#
#     print("\nPer-dimension stats:")
#     for d in range(y_true.shape[1]):
#         print(
#             f"  dim {d:>2}: "
#             f"MSE={per_dim_mse[d]:.8f}  "
#             f"MAE={per_dim_mae[d]:.8f}  "
#             f"P95={per_dim_p95[d]:.8f}  "
#             f"P99={per_dim_p99[d]:.8f}  "
#             f"MAX={per_dim_max[d]:.8f}"
#         )
#
#     # Worst samples
#     worst_score, worst_dim = abs_err.max(dim=1)
#     top_k = min(top_k, n_eval)
#     worst_idx = torch.topk(worst_score, k=top_k).indices
#
#     worst_samples = []
#
#     print(f"\nTop-{top_k} worst samples by max component error:")
#     for rank, i in enumerate(worst_idx.tolist(), start=1):
#         i_t = int(i)
#         dim_i = int(worst_dim[i_t].item())
#
#         x_i = x[i_t].detach().cpu().numpy()
#         u_i = u[i_t].detach().cpu().numpy()
#         y_true_i = y_true[i_t].detach().cpu().numpy()
#         y_pred_i = y_pred[i_t].detach().cpu().numpy()
#         abs_err_i = abs_err[i_t].detach().cpu().numpy()
#
#         print(f"\n  [{rank}] sample idx in eval batch = {i_t}")
#         print(f"      worst dim      = {dim_i}")
#         print(f"      max|err|       = {worst_score[i_t].item():.8f}")
#         print(f"      state x        = {np.array2string(x_i, precision=5, suppress_small=False)}")
#         print(f"      action u       = {np.array2string(u_i, precision=5, suppress_small=False)}")
#         print(f"      target         = {np.array2string(y_true_i, precision=5, suppress_small=False)}")
#         print(f"      prediction     = {np.array2string(y_pred_i, precision=5, suppress_small=False)}")
#         print(f"      abs error      = {np.array2string(abs_err_i, precision=5, suppress_small=False)}")
#
#         worst_samples.append({
#             "rank": rank,
#             "sample_idx_in_eval_batch": i_t,
#             "worst_dim": dim_i,
#             "max_abs_err": float(worst_score[i_t].item()),
#             "x": x_i.tolist(),
#             "u": u_i.tolist(),
#             "target": y_true_i.tolist(),
#             "prediction": y_pred_i.tolist(),
#             "abs_error": abs_err_i.tolist(),
#         })
#
#     return {
#         "mse": float(mse),
#         "mae": float(mae),
#         "p95": float(p95),
#         "p99": float(p99),
#         "max_err": float(max_err),
#         "per_dim_mse": per_dim_mse.tolist(),
#         "per_dim_mae": per_dim_mae.tolist(),
#         "per_dim_p95": per_dim_p95.tolist(),
#         "per_dim_p99": per_dim_p99.tolist(),
#         "per_dim_max": per_dim_max.tolist(),
#         "worst_samples": worst_samples,
#     }
#
#
# # -----------------------------
# # Load data
# # -----------------------------
# data_path = "../../data/robot_arm_data.npz"
# print(f"Loading data from {data_path}...")
# data_npz = np.load(data_path)
# data = {key: data_npz[key] for key in data_npz.keys()}
#
# print("\nLoaded dataset keys:")
# for k, v in data.items():
#     print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
#
#
# # -----------------------------
# # Create dynamics model
# # -----------------------------
# # Following the other benchmark scripts:
# # two hidden layers of 32 neurons.
# dynamics_hidden_sizes = [32, 32]
#
# f_model = create_mlp(
#     RobotArm.x_dim + RobotArm.u_dim,
#     RobotArm.x_dim,
#     dynamics_hidden_sizes
# )
#
# print("\nDynamics model architecture:")
# print(f"  Input dim: {RobotArm.x_dim + RobotArm.u_dim}")
# print(f"  Hidden sizes: {dynamics_hidden_sizes}")
# print(f"  Output dim: {RobotArm.x_dim}  (predicting dx)")
# print(f"  Total parameters: {sum(p.numel() for p in f_model.parameters())}")
#
#
# # -----------------------------
# # Train dynamics model
# # -----------------------------
# print("\nTraining dynamics model...")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
#
# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
#
# # State-action space size for APA noise scaling
# xu_space_size = np.concatenate([
#     RobotArm.x_high - RobotArm.x_low,
#     RobotArm.u_high - RobotArm.u_low
# ]).astype(np.float32)
#
# log_path = train_dynamics(
#     data=data,
#     f_model=f_model,
#     closed_loop=False,
#     lr=1e-3,
#     batch_size=256,
#     epoch_num=100,
#     weight_decay=0.0,
#     penalty="APA",
#     noise_scale=0.1,
#     space_size=xu_space_size,
#     apa_coef=1e-4,
#     log_dir="../../logs/robot_arm/",
#     device=device,
#     seed=seed,
#     task="robot_arm"
# )
#
#
# # -----------------------------
# # Final save (robust)
# # -----------------------------
# models_dir = "../../models/robot_arm/"
# ensure_dir(models_dir)
#
# f_path = os.path.join(models_dir, "f.pth")
# atomic_torch_save(f_model.state_dict(), f_path)
#
#
# # -----------------------------
# # Quick + detailed post-training sanity check
# # -----------------------------
# print("\nRunning detailed post-training sanity check...")
#
# # We try to infer common key names used in the dataset
# possible_x_keys = ["x", "X", "state", "states"]
# possible_u_keys = ["u", "U", "action", "actions"]
# possible_dx_keys = ["dx", "DX", "delta", "dynamics_target"]
# possible_xnext_keys = ["x_next", "X_next", "next_state", "next_states", "y", "Y"]
#
# x_key = next((k for k in possible_x_keys if k in data), None)
# u_key = next((k for k in possible_u_keys if k in data), None)
# dx_key = next((k for k in possible_dx_keys if k in data), None)
# xnext_key = next((k for k in possible_xnext_keys if k in data), None)
#
# sanity_metrics = {}
#
# if x_key is not None and u_key is not None and (dx_key is not None or xnext_key is not None):
#     X = data[x_key]
#     U = data[u_key]
#
#     # Normalize shape to [N, dim]
#     if X.ndim != 2:
#         raise ValueError(f"Expected X to be 2D, got shape {X.shape}")
#     if U.ndim != 2:
#         raise ValueError(f"Expected U to be 2D, got shape {U.shape}")
#
#     if X.shape[1] != RobotArm.x_dim and X.shape[0] == RobotArm.x_dim:
#         X = X.T
#     if U.shape[1] != RobotArm.u_dim and U.shape[0] == RobotArm.u_dim:
#         U = U.T
#
#     DX = None
#     X_next = None
#
#     if dx_key is not None:
#         DX = data[dx_key]
#         if DX.ndim != 2:
#             raise ValueError(f"Expected DX to be 2D, got shape {DX.shape}")
#         if DX.shape[1] != RobotArm.x_dim and DX.shape[0] == RobotArm.x_dim:
#             DX = DX.T
#
#     if xnext_key is not None:
#         X_next = data[xnext_key]
#         if X_next.ndim != 2:
#             raise ValueError(f"Expected X_next to be 2D, got shape {X_next.shape}")
#         if X_next.shape[1] != RobotArm.x_dim and X_next.shape[0] == RobotArm.x_dim:
#             X_next = X_next.T
#
#     eval_dataset = {
#         "x": X.astype(np.float32),
#         "u": U.astype(np.float32),
#         "dx": DX.astype(np.float32) if DX is not None else None,
#         "x_next": X_next.astype(np.float32) if X_next is not None else None,
#     }
#
#     f_model = f_model.to(device).eval()
#
#     detailed_stats = sanity_check_dynamics_detailed(
#         f_model=f_model,
#         dataset=eval_dataset,
#         device=device,
#         n_eval=5000,
#         target_source="dx",
#         top_k=10,
#     )
#
#     sanity_metrics = {
#         "n_eval": min(5000, eval_dataset["x"].shape[0]),
#         "target_source": dx_key if dx_key is not None else xnext_key,
#         "mse_dx": detailed_stats["mse"],
#         "mae_dx": detailed_stats["mae"],
#         "p95_abs_err_dx": detailed_stats["p95"],
#         "p99_abs_err_dx": detailed_stats["p99"],
#         "max_abs_err_dx": detailed_stats["max_err"],
#         "per_dim_mse_dx": detailed_stats["per_dim_mse"],
#         "per_dim_mae_dx": detailed_stats["per_dim_mae"],
#         "per_dim_p95_dx": detailed_stats["per_dim_p95"],
#         "per_dim_p99_dx": detailed_stats["per_dim_p99"],
#         "per_dim_max_dx": detailed_stats["per_dim_max"],
#         "worst_samples": detailed_stats["worst_samples"],
#         "x_key": x_key,
#         "u_key": u_key,
#     }
#
# else:
#     print("Could not run sanity check automatically: dataset keys not recognized.")
#
#
# # -----------------------------
# # Save summary
# # -----------------------------
# summary_path = os.path.join(models_dir, "f_train_summary.json")
# summary = {
#     "system": "robot_arm",
#     "device": device,
#     "data_path": os.path.abspath(data_path),
#     "model_path": os.path.abspath(f_path),
#     "hidden_sizes": dynamics_hidden_sizes,
#     "seed": seed,
#     "train_config": {
#         "lr": 1e-3,
#         "batch_size": 256,
#         "epoch_num": 100,
#         "weight_decay": 0.0,
#         "penalty": "APA",
#         "noise_scale": 0.1,
#         "apa_coef": 1e-4,
#         "closed_loop": False,
#     },
#     "sanity_metrics": sanity_metrics,
#     "log_path": os.path.abspath(log_path),
# }
# save_json(summary, summary_path)
#
#
# print("\nTraining complete!")
# print("Model saved to:", os.path.abspath(f_path))
# print("Summary saved to:", os.path.abspath(summary_path))
# print("Logs saved to:", os.path.abspath(log_path))
# print("\nTo view training curves, run:")
# print(f"  tensorboard --logdir={os.path.abspath(log_path)}")