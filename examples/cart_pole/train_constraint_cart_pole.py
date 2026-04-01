"""
Train constraint model for the CartPole system.
"""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
from systems.cart_pole import CartPole
from cersyve import create_mlp, train_constraint

# Create constraint model
constraint_hidden_sizes = [32, 32]
h_model = create_mlp(CartPole.x_dim, 1, constraint_hidden_sizes)

print("Constraint model architecture:")
print(f"  Input dim: {CartPole.x_dim}")
print(f"  Hidden sizes: {constraint_hidden_sizes}")
print(f"  Output dim: 1")
print(f"  Total parameters: {sum(p.numel() for p in h_model.parameters())}")

# Train constraint model
print("\nTraining constraint model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set random seed for reproducibility
seed = 42

log_path = train_constraint(
    x_low=CartPole.x_low,
    x_high=CartPole.x_high,
    h_model=h_model,
    constraint=CartPole.constraint,
    lr=3e-4,
    batch_size=256,
    n_iter=150000,
    log_dir='../../logs/cart_pole/',
    device=device,
    seed=seed,
    task="cart_pole"
)

# -----------------------------
# Post-training evaluation
# -----------------------------


print("\nEvaluating trained constraint model...")

# metti il modello in eval
h_model = h_model.to(device).eval()

# numero di campioni per la valutazione
n_eval = 100000

# campionamento uniforme nel box di stato
x_eval = np.random.uniform(
    CartPole.x_low[:, None],
    CartPole.x_high[:, None],
    size=(CartPole.x_dim, n_eval)
).astype(np.float32)

x_eval_t = torch.from_numpy(x_eval).to(device)

# predizione del modello h
with torch.no_grad():
    h_pred = h_model(x_eval_t.T).T.cpu().numpy()[0]

# constraint vera
h_true = CartPole.constraint(x_eval)

# forza tutto a vettori 1D
h_pred = np.asarray(h_pred).reshape(-1)
h_true = np.asarray(h_true).reshape(-1)

true_safe = (h_true <= 0)
pred_safe = (h_pred <= 0)

# metriche
true_safe_rate = float(true_safe.mean())
pred_safe_rate = float(pred_safe.mean())
accuracy = float((true_safe == pred_safe).mean())

tp = int(( true_safe &  pred_safe).sum())
tn = int((~true_safe & ~pred_safe).sum())
fp = int((~true_safe &  pred_safe).sum())
fn = int(( true_safe & ~pred_safe).sum())

fp_rate = fp / max(int((~true_safe).sum()), 1)
fn_rate = fn / max(int(( true_safe).sum()), 1)

print("\n[CONSTRAINT EVAL] true constraint stats:")
print(f"  true safe rate:      {true_safe_rate:.6f}")

print("\n[CONSTRAINT EVAL] h_model stats:")
print(f"  predicted safe rate: {pred_safe_rate:.6f}")
print(f"  accuracy:            {accuracy:.6f}")
print(f"  false positive rate: {fp_rate:.6f}")
print(f"  false negative rate: {fn_rate:.6f}")

print("\n[CONSTRAINT EVAL] confusion counts:")
print(f"  TP (safe -> safe):     {tp}")
print(f"  TN (unsafe -> unsafe): {tn}")
print(f"  FP (unsafe -> safe):   {fp}")
print(f"  FN (safe -> unsafe):   {fn}")

print("\n[CONSTRAINT EVAL] h_model output stats:")
print(
    f"  min={float(h_pred.min()):.6f}, "
    f"p1={float(np.percentile(h_pred, 1)):.6f}, "
    f"p10={float(np.percentile(h_pred, 10)):.6f}, "
    f"p50={float(np.percentile(h_pred, 50)):.6f}, "
    f"p90={float(np.percentile(h_pred, 90)):.6f}, "
    f"p99={float(np.percentile(h_pred, 99)):.6f}, "
    f"max={float(h_pred.max()):.6f}"
)

h_pred_safe = h_pred[true_safe]
h_pred_unsafe = h_pred[~true_safe]

if len(h_pred_safe) > 0:
    print("\n[CONSTRAINT EVAL] h_model on TRUE SAFE states:")
    print(
        f"  mean={float(h_pred_safe.mean()):.6f}, "
        f"p50={float(np.percentile(h_pred_safe, 50)):.6f}, "
        f"p90={float(np.percentile(h_pred_safe, 90)):.6f}"
    )

if len(h_pred_unsafe) > 0:
    print("\n[CONSTRAINT EVAL] h_model on TRUE UNSAFE states:")
    print(
        f"  mean={float(h_pred_unsafe.mean()):.6f}, "
        f"p10={float(np.percentile(h_pred_unsafe, 10)):.6f}, "
        f"p50={float(np.percentile(h_pred_unsafe, 50)):.6f}"
    )
print("\nTraining complete!")
print("Model saved to: models/cart_pole/h.pth")
print(f"Logs saved to: {log_path}")
print("\nTo view training curves, run:")
print(f"  tensorboard --logdir={log_path}")