"""
Train constraint model for the RobotArm system
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

from systems.robot_arm import RobotArm
from cersyve import create_mlp, train_constraint


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
# Evaluation
# -----------------------------
def evaluate_constraint_model(
    h_model,
    n_eval: int = 100000,
    device: str = "cpu",
    label: str = "CONSTRAINT_EVAL"
):
    print(f"\nEvaluating trained constraint model [{label}]...")

    h_model = h_model.to(device).eval()

    # Uniform evaluation over the full benchmark state box
    x_eval = np.random.uniform(
        RobotArm.x_low[:, None],
        RobotArm.x_high[:, None],
        size=(RobotArm.x_dim, n_eval)
    ).astype(np.float32)

    x_eval_t = torch.from_numpy(x_eval.T).to(device)

    with torch.no_grad():
        h_pred = h_model(x_eval_t).cpu().numpy().reshape(-1)

    h_true = RobotArm.constraint(x_eval).reshape(-1)

    true_safe = (h_true <= 0)
    pred_safe = (h_pred <= 0)

    true_safe_rate = float(true_safe.mean())
    pred_safe_rate = float(pred_safe.mean())
    accuracy = float((true_safe == pred_safe).mean())

    tp = int(( true_safe &  pred_safe).sum())
    tn = int((~true_safe & ~pred_safe).sum())
    fp = int((~true_safe &  pred_safe).sum())
    fn = int(( true_safe & ~pred_safe).sum())

    fp_rate = fp / max(int((~true_safe).sum()), 1)
    fn_rate = fn / max(int(( true_safe).sum()), 1)

    print(f"\n[{label}] true constraint stats:")
    print(f"  true safe rate:      {true_safe_rate:.6f}")

    print(f"\n[{label}] h_model stats:")
    print(f"  predicted safe rate: {pred_safe_rate:.6f}")
    print(f"  accuracy:            {accuracy:.6f}")
    print(f"  false positive rate: {fp_rate:.6f}")
    print(f"  false negative rate: {fn_rate:.6f}")

    print(f"\n[{label}] confusion counts:")
    print(f"  TP (safe -> safe):     {tp}")
    print(f"  TN (unsafe -> unsafe): {tn}")
    print(f"  FP (unsafe -> safe):   {fp}")
    print(f"  FN (safe -> unsafe):   {fn}")

    print(f"\n[{label}] h_model output stats:")
    print(
        f"  min={float(h_pred.min()):.6f}, "
        f"p1={float(np.percentile(h_pred, 1)):.6f}, "
        f"p10={float(np.percentile(h_pred, 10)):.6f}, "
        f"p50={float(np.percentile(h_pred, 50)):.6f}, "
        f"p90={float(np.percentile(h_pred, 90)):.6f}, "
        f"p99={float(np.percentile(h_pred, 99)):.6f}, "
        f"max={float(h_pred.max()):.6f}"
    )

    return {
        "true_safe_rate": true_safe_rate,
        "pred_safe_rate": pred_safe_rate,
        "accuracy": accuracy,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# -----------------------------
# Create constraint model
# -----------------------------
constraint_hidden_sizes = [16]

h_model = create_mlp(
    RobotArm.x_dim,
    1,
    constraint_hidden_sizes
)

print("Constraint model architecture:")
print(f"  Input dim: {RobotArm.x_dim}")
print(f"  Hidden sizes: {constraint_hidden_sizes}")
print(f"  Output dim: 1")
print(f"  Total parameters: {sum(p.numel() for p in h_model.parameters())}")


# -----------------------------
# Train constraint model
# -----------------------------
print("\nTraining constraint model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

log_path = train_constraint(
    x_low=RobotArm.x_low,
    x_high=RobotArm.x_high,
    h_model=h_model,
    constraint=RobotArm.constraint,
    lr=3e-4,
    batch_size=256,
    n_iter=120000,
    log_dir="../../logs/robot_arm/",
    device=device,
    seed=seed,
    task="robot_arm"
)


# -----------------------------
# Evaluation
# -----------------------------
metrics = evaluate_constraint_model(
    h_model=h_model,
    n_eval=100000,
    device=device,
    label="FINAL"
)


# -----------------------------
# Save
# -----------------------------
models_dir = "../../models/robot_arm/"
ensure_dir(models_dir)

h_path = os.path.join(models_dir, "h.pth")
atomic_torch_save(h_model.state_dict(), h_path)

print("\nTraining complete!")
print("Model saved to:", os.path.abspath(h_path))
print(f"Logs saved to: {os.path.abspath(log_path)}")

print("\nTo view training curves, run:")
print(f"  tensorboard --logdir={os.path.abspath(log_path)}")