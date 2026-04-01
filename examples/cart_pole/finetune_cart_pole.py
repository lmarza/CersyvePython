"""
Finetune value function for the CartPole system (ALL FLOAT32 + fixed save).
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
from systems.cart_pole import CartPole
from cersyve import create_mlp, create_closed_loop_dynamics_model, finetune_value


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def atomic_torch_save(state_dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)


# -----------------------------
# Load data (FORCE float32)
# -----------------------------
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

# Make sure bounds are float32
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
    f_model, pi_model, data, x_low, x_high, CartPole.u_dim
).to(device).float().eval()

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
# Create and load pretrained value function V (FLOAT32)
# -----------------------------
value_hidden_sizes = [64, 64]
V_model = create_mlp(CartPole.x_dim, 1, value_hidden_sizes)
V_pretrain_path = os.path.join(models_dir, "V_pretrain.pth")
print(f"Loading pretrained value function from {V_pretrain_path}...")
V_model.load_state_dict(torch.load(V_pretrain_path, map_location=device))
V_model = V_model.to(device).float().train()

print("\nValue function model architecture:")
print(f"  Input dim: {CartPole.x_dim}")
print(f"  Hidden sizes: {value_hidden_sizes}")
print("  Output dim: 1")
print(f"  Total parameters: {sum(p.numel() for p in V_model.parameters())}")

print("\nFinetuning value function...")
seed = 42

# --- quick diagnostics ---
N = 20000
x_dbg = np.random.uniform(x_low[:, None], x_high[:, None], size=(CartPole.x_dim, N)).astype(np.float32)
x_dbg_t = torch.from_numpy(x_dbg).to(device)

with torch.no_grad():
    v_dbg = V_model(x_dbg_t.T).T.cpu().numpy()[0]
    h_dbg = h_model(x_dbg_t.T).T.cpu().numpy()[0]

print("\n[DEBUG] V stats:",
      "min", float(v_dbg.min()),
      "p1", float(np.percentile(v_dbg, 1)),
      "p50", float(np.percentile(v_dbg, 50)),
      "p99", float(np.percentile(v_dbg, 99)),
      "max", float(v_dbg.max()))
print("[DEBUG] V<=0 rate:", float((v_dbg <= 0).mean()))

print("[DEBUG] h stats:",
      "min", float(h_dbg.min()),
      "p1", float(np.percentile(h_dbg, 1)),
      "p50", float(np.percentile(h_dbg, 50)),
      "p99", float(np.percentile(h_dbg, 99)),
      "max", float(h_dbg.max()))
print("[DEBUG] h<=0 rate:", float((h_dbg <= 0).mean()))

# boundary band used by finetune:
band = (v_dbg > -0.25) & (v_dbg <= 1e-3)
print("[DEBUG] boundary band count:", int(band.sum()), "over", N)

# # --- SHIFT V so that ~30% of points are <= 0 (bootstrap boundary) ---
# target_q = 30  # percento che vuoi <=0
# delta = float(np.percentile(v_dbg, target_q))
#
# with torch.no_grad():
#     last_linear = None
#     for m in V_model.modules():
#         if isinstance(m, torch.nn.Linear):
#             last_linear = m
#     assert last_linear is not None
#     last_linear.bias.data -= torch.tensor(delta, device=device, dtype=last_linear.bias.dtype)
#
# print(f"[DEBUG] shifted V bias by {delta:.6f} (target {target_q}th percentile)")

log_path = finetune_value(
    V_model=V_model,
    f_pi_model=f_pi_model,
    h_model=h_model,
    x_low=x_low,
    x_high=x_high,
    lr=5e-5,
    max_iter=200000,
    search_size=12000,
    bnd_ratio=0.1,
    bnd_ratio_avg=0.9,
    min_bnd_ratio=0.01,
    max_bnd_ratio=1.0,
    bnd_eps=0.2,
    search_method="PGD-B",
    pgd_step=40,
    pgd_eps=0.2,
    backtrack_step=30,
    length_discount=0.8,
    direct_discount=0.5,
    tol=5e-3,
    capacity=30000,
    sample_size=512,
    search_stop=8000,
    replay=5,
    max_skip=3000,
    reg_method="ESR",
    eps_h=0.02,
    eps_v=0.1,
    reg_coef=1.0,
    log_dir=logs_dir,
    eval_every=50,
    save_every=2000,
    save_path=os.path.join(models_dir, "V_finetune.pth"),
    device=device,
    seed=seed
)
# -----------------------------
# FORCED SAVE (FIXED: save V_model, not h_model)
# -----------------------------
V_finetune_path = os.path.join(models_dir, "V_finetune.pth")
atomic_torch_save(V_model.state_dict(), V_finetune_path)
print("FORCED SAVE:", os.path.abspath(V_finetune_path))

print("\nFinetuning complete!")
print("Model saved to:", os.path.abspath(V_finetune_path))
print("Logs saved to:", os.path.abspath(log_path))
print("\nTo view training curves, run:")
print(f"  tensorboard --logdir={os.path.abspath(log_path)}")