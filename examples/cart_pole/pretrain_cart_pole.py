"""
Example: Pretrain value function for the CartPole system
(ALL FLOAT32 + diagnostics + post-pretrain bias calibration).
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
from systems.cart_pole import CartPole
from cersyve import create_mlp, create_closed_loop_dynamics_model, pretrain_value
from cersyve.utils import uniform


# -----------------------------
# Robust saving utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def atomic_torch_save(state_dict, path: str) -> None:
    """Save to tmp then rename (atomic on POSIX)."""
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
    """
    Quick Monte Carlo diagnostics on the current value function.
    """
    V_model.eval()
    h_model.eval()

    X = uniform(x_low, x_high, n_samples).astype(np.float32)   # [x_dim, N]
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


def calibrate_value_bias(
    V_model,
    x_low,
    x_high,
    device="cpu",
    n_samples=100000,
    target_q=12.0
):
    """
    Shift the last linear bias so that about `target_q` percent of random states
    satisfy V(x) <= 0. This is a practical re-centering step before finetuning.
    """
    V_model.eval()

    X = uniform(x_low, x_high, n_samples).astype(np.float32)
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        V = V_model(X_t.T).T.cpu().numpy()[0]

    delta = float(np.percentile(V, target_q))

    last_linear = None
    for m in V_model.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m

    if last_linear is None:
        raise RuntimeError("No final Linear layer found in V_model.")

    with torch.no_grad():
        last_linear.bias.data -= torch.tensor(delta, device=device, dtype=last_linear.bias.dtype)

    print(f"\n[CALIBRATION] Shifted final bias by {delta:.6f} to target ~{target_q:.1f}% with V<=0")

    V_model.train()


# -----------------------------
# Main
# -----------------------------
# Load data (force float32 everywhere)
data_path = "../../data/cart_pole_data.npz"
print(f"Loading data from {data_path}...")
data_npz = np.load(data_path)

data = {}
for k in data_npz.keys():
    arr = data_npz[k]
    if isinstance(arr, np.ndarray) and arr.dtype.kind == "f":
        arr = arr.astype(np.float32)
    data[k] = arr

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Ensure log/model dirs exist
logs_dir = "../../logs/cart_pole/"
models_dir = "../../models/cart_pole/"
ensure_dir(logs_dir)
ensure_dir(models_dir)

x_low = CartPole.x_low.astype(np.float32)
x_high = CartPole.x_high.astype(np.float32)

# -----------------------------
# Load dynamics model f (FLOAT32)
# -----------------------------
dynamics_hidden_sizes = [64, 64]
f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, dynamics_hidden_sizes)
f_model_path = os.path.join(models_dir, "f.pth")
print(f"Loading dynamics model from {f_model_path}...")
f_model.load_state_dict(torch.load(f_model_path, map_location=device))
f_model = f_model.to(device).float().eval()

# -----------------------------
# Create closed-loop dynamics model f_pi (FLOAT32)
# -----------------------------
pi_model = CartPole.create_policy()
pi_model = pi_model.to(device).float().eval()

f_pi_model = create_closed_loop_dynamics_model(
    f_model,
    pi_model,
    data,
    x_low,
    x_high,
    CartPole.u_dim
)
f_pi_model = f_pi_model.to(device).float().eval()
print("Created closed-loop dynamics model")

# -----------------------------
# Load constraint model h (FLOAT32)
# -----------------------------
constraint_hidden_sizes = [32, 32]
h_model = create_mlp(CartPole.x_dim, 1, constraint_hidden_sizes)
h_model_path = os.path.join(models_dir, "h.pth")
print(f"Loading constraint model from {h_model_path}...")
h_model.load_state_dict(torch.load(h_model_path, map_location=device))
h_model = h_model.to(device).float().eval()

# -----------------------------
# Create value function model V (FLOAT32)
# -----------------------------
value_hidden_sizes = [64, 64]
V_model = create_mlp(CartPole.x_dim, 1, value_hidden_sizes)
V_model = V_model.to(device).float().train()

print("\nValue function model architecture:")
print(f"  Input dim: {CartPole.x_dim}")
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
    lr=1e-4,                 # was 3e-4
    batch_size=1024,         # was 512
    iter_num=180000,         # was 150000
    weight_decay=0.0,
    penalty="APA",
    noise_scale=0.03,        # was 0.1
    space_size=(x_high - x_low).astype(np.float32),
    apa_coef=1e-4,           # was 1e-3
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

# -----------------------------
# Post-pretrain calibration
# -----------------------------
calibrate_value_bias(
    V_model=V_model,
    x_low=x_low,
    x_high=x_high,
    device=device,
    n_samples=100000,
    target_q=12.0
)

# Diagnostics after calibration
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

f_snapshot = os.path.join(models_dir, "f_snapshot.pth")
h_snapshot = os.path.join(models_dir, "h_snapshot.pth")
atomic_torch_save(f_model.state_dict(), f_snapshot)
atomic_torch_save(h_model.state_dict(), h_snapshot)
print("Snapshot f:", os.path.abspath(f_snapshot))
print("Snapshot h:", os.path.abspath(h_snapshot))

print("\nPretraining complete!")
print("Model saved to:", os.path.abspath(V_pretrain_path))
print("Logs saved to:", os.path.abspath(log_path))
print("\nTo view training curves, run:")
print(f"  tensorboard --logdir={os.path.abspath(log_path)}")