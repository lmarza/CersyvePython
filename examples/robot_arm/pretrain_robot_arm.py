"""
Pretrain value function for the RobotArm system
(ALL FLOAT32 + diagnostics + post-pretrain bias calibration).
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np

from systems.robot_arm import RobotArm
from cersyve import create_mlp, pretrain_value
from cersyve.utils import uniform


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


# -----------------------------
# Diagnostics
# -----------------------------
def print_value_diagnostics(
    V_model,
    h_model,
    x_low,
    x_high,
    device="cpu",
    n_samples=50000,
    boundary_eps=0.25,
    label="DIAG"
):
    V_model.eval()
    h_model.eval()

    X = uniform(x_low, x_high, n_samples).astype(np.float32)
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        V = V_model(X_t.T).T.cpu().numpy()[0]
        H = h_model(X_t.T).T.cpu().numpy()[0]

    print(f"\n[{label}] V stats:")
    print(
        "  min", float(V.min()),
        "p1", float(np.percentile(V, 1)),
        "p10", float(np.percentile(V, 10)),
        "p50", float(np.percentile(V, 50)),
        "p90", float(np.percentile(V, 90)),
        "p99", float(np.percentile(V, 99)),
        "max", float(V.max()),
    )
    print(f"[{label}] V<=0 rate: {float((V <= 0).mean()):.6f}")
    print(f"[{label}] boundary band (-{boundary_eps} < V <= 0): {float(((V > -boundary_eps) & (V <= 0)).mean()):.6f}")

    print(f"\n[{label}] h stats:")
    print(
        "  min", float(H.min()),
        "p1", float(np.percentile(H, 1)),
        "p10", float(np.percentile(H, 10)),
        "p50", float(np.percentile(H, 50)),
        "p90", float(np.percentile(H, 90)),
        "p99", float(np.percentile(H, 99)),
        "max", float(H.max()),
    )
    print(f"[{label}] h<=0 rate: {float((H <= 0).mean()):.6f}")

    V_model.train()

# -----------------------------
# Main
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

logs_dir = "../../logs/robot_arm/"
models_dir = "../../models/robot_arm/"
ensure_dir(logs_dir)
ensure_dir(models_dir)

x_low = RobotArm.x_low.astype(np.float32)
x_high = RobotArm.x_high.astype(np.float32)

# -----------------------------
# Exact closed-loop dynamics model f_pi
# -----------------------------
pi_model = RobotArm.create_policy().to(device).float().eval()
f_pi_model = RobotArm.create_closed_loop_dynamics_model().to(device).float().eval()
print("Created exact closed-loop dynamics model")

# -----------------------------
# Load constraint model h (FLOAT32)
# -----------------------------
constraint_hidden_sizes = [16]
h_model = create_mlp(RobotArm.x_dim, 1, constraint_hidden_sizes)
h_model_path = os.path.join(models_dir, "h.pth")
print(f"Loading constraint model from {h_model_path}...")
h_model.load_state_dict(torch.load(h_model_path, map_location=device))
h_model = h_model.to(device).float().eval()

# -----------------------------
# Create value function model V (FLOAT32)
# -----------------------------
value_hidden_sizes = [32, 32]
V_model = create_mlp(RobotArm.x_dim, 1, value_hidden_sizes)
V_model = V_model.to(device).float().train()

print("\nValue function model architecture:")
print(f"  Input dim: {RobotArm.x_dim}")
print(f"  Hidden sizes: {value_hidden_sizes}")
print("  Output dim: 1")
print(f"  Total parameters: {sum(p.numel() for p in V_model.parameters())}")

# -----------------------------
# Initial diagnostics
# -----------------------------
print_value_diagnostics(
    V_model=V_model,
    h_model=h_model,
    x_low=x_low,
    x_high=x_high,
    device=device,
    n_samples=50000,
    boundary_eps=0.25,
    label="BEFORE_PRETRAIN"
)

# -----------------------------
# Pretrain value function
# -----------------------------
print("\nPretraining value function...")
seed = 42

log_path = pretrain_value(
    V_model=V_model,
    f_pi_model=f_pi_model,
    h_model=h_model,
    x_low=x_low,
    x_high=x_high,
    gamma=0.9,
    lr=3e-4,
    batch_size=512,
    iter_num=180000,
    weight_decay=0.0,
    penalty="APA",
    noise_scale=0.03,
    space_size=(x_high - x_low).astype(np.float32),
    apa_coef=1e-4,
    log_dir=logs_dir,
    device=device,
    seed=seed
)

# -----------------------------
# Diagnostics after pretrain
# -----------------------------
print_value_diagnostics(
    V_model=V_model,
    h_model=h_model,
    x_low=x_low,
    x_high=x_high,
    device=device,
    n_samples=50000,
    boundary_eps=0.25,
    label="AFTER_PRETRAIN"
)

print_value_diagnostics(
    V_model=V_model,
    h_model=h_model,
    x_low=x_low,
    x_high=x_high,
    device=device,
    n_samples=50000,
    boundary_eps=0.25,
    label="AFTER_CALIBRATION"
)

# -----------------------------
# FORCED SAVE
# -----------------------------
V_pretrain_path = os.path.join(models_dir, "V_pretrain.pth")
atomic_torch_save(V_model.state_dict(), V_pretrain_path)
print("FORCED SAVE V_pretrain:", os.path.abspath(V_pretrain_path))

pi_snapshot = os.path.join(models_dir, "pi_exact_snapshot.pth")
f_pi_snapshot = os.path.join(models_dir, "f_pi_exact_snapshot.pth")
h_snapshot = os.path.join(models_dir, "h_snapshot.pth")

atomic_torch_save(pi_model.state_dict(), pi_snapshot)
atomic_torch_save(f_pi_model.state_dict(), f_pi_snapshot)
atomic_torch_save(h_model.state_dict(), h_snapshot)

print("Snapshot pi:", os.path.abspath(pi_snapshot))
print("Snapshot f_pi:", os.path.abspath(f_pi_snapshot))
print("Snapshot h:", os.path.abspath(h_snapshot))

print("\nPretraining complete!")
print("Model saved to:", os.path.abspath(V_pretrain_path))
print("Logs saved to:", os.path.abspath(log_path))
print("\nTo view training curves, run:")
print(f"  tensorboard --logdir={os.path.abspath(log_path)}")