"""
Compare pretrained vs finetuned V on CartPole, compute TFR and show how to check < 0.404.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from systems.cart_pole import CartPole
from cersyve import create_mlp, create_closed_loop_dynamics_model
from cersyve.utils import uniform

def compute_tfr_metrics(V_model, h_model, f_pi_model, n_samples=200_000, horizon=100, device="cpu"):
    """
    tfr (in your definition) = recall = P(V<=0 | truly feasible) over truly feasible states.
    also returns truly_feasible_rate = P(truly feasible).
    and precision = P(truly feasible | V<=0).
    """
    V_model.eval()
    h_model.eval()
    f_pi_model.eval()

    X = uniform(CartPole.x_low, CartPole.x_high, n_samples)          # [4,N]
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        V = V_model(X_t.T).squeeze(1)                                # [N]
        pred_safe = (V <= 0)

        true_safe = torch.ones(n_samples, dtype=torch.bool, device=device)
        Xcur = X_t
        for _ in range(horizon):
            hcur = h_model(Xcur.T).squeeze(1)
            true_safe &= (hcur <= 0)
            Xcur = f_pi_model(Xcur)

        tp = (pred_safe & true_safe).sum().item()
        fn = ((~pred_safe) & true_safe).sum().item()
        fp = (pred_safe & (~true_safe)).sum().item()
        n_true = true_safe.sum().item()
        n_pred = pred_safe.sum().item()

    recall_tfr = tp / max(tp + fn, 1)          # your “TFR”
    precision = tp / max(tp + fp, 1)
    true_feasible_rate = n_true / n_samples

    return {
        "tfr_recall": recall_tfr,
        "precision": precision,
        "true_feasible_rate": true_feasible_rate,
        "n_true": n_true,
        "n_pred_safe": n_pred,
        "tp": tp, "fn": fn, "fp": fp
    }

def main(finetuned_path="../../models/cart_pole/V_finetune_enum_milp.pth",
         threshold=0.404):
    os.makedirs("plots", exist_ok=True)
    device = "cpu"

    models_dir = "../../models/cart_pole/"
    data_path = "../../data/cart_pole_data.npz"
    data_npz = np.load(data_path)
    data = {k: data_npz[k] for k in data_npz.keys()}

    f_model = create_mlp(CartPole.x_dim + CartPole.u_dim, CartPole.x_dim, [64, 64])
    f_model.load_state_dict(torch.load(os.path.join(models_dir, "f.pth"), map_location=device))
    f_model.eval()

    pi_model = CartPole.create_policy().eval()
    f_pi_model = create_closed_loop_dynamics_model(f_model, pi_model, data, CartPole.x_low, CartPole.x_high, CartPole.u_dim)
    f_pi_model.eval()

    h_model = create_mlp(CartPole.x_dim, 1, [32, 32])
    h_model.load_state_dict(torch.load(os.path.join(models_dir, "h.pth"), map_location=device))
    h_model.eval()

    V_pre = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_pre.load_state_dict(torch.load(os.path.join(models_dir, "V_pretrain.pth"), map_location=device))
    V_pre.eval()

    V_fine = create_mlp(CartPole.x_dim, 1, [64, 64])
    V_fine.load_state_dict(torch.load(finetuned_path, map_location=device))
    V_fine.eval()

    # Metrics
    m_pre  = compute_tfr_metrics(V_pre,  h_model, f_pi_model, device=device)
    m_fine = compute_tfr_metrics(V_fine, h_model, f_pi_model, device=device)

    print("\n=== CartPole metrics (100-step) ===")
    print("Pretrain:  tfr(recall)={:.3f}  precision={:.3f}  true_feasible_rate={:.3f}".format(
        m_pre["tfr_recall"], m_pre["precision"], m_pre["true_feasible_rate"]))
    print("Finetune:  tfr(recall)={:.3f}  precision={:.3f}  true_feasible_rate={:.3f}".format(
        m_fine["tfr_recall"], m_fine["precision"], m_fine["true_feasible_rate"]))

    # How to check "< 0.404" correctly:
    # - If 0.404 is your target/threshold for the recall, compare with tfr_recall.
    # - If 0.404 is the fraction of truly feasible states in the domain, compare with true_feasible_rate.
    print("\n=== Check against threshold {} ===".format(threshold))
    print("Finetune tfr(recall) < threshold ? ->", m_fine["tfr_recall"] < threshold)
    print("Finetune true_feasible_rate < threshold ? ->", m_fine["true_feasible_rate"] < threshold)

    # If you want to hard-fail when below threshold:
    # assert m_fine["tfr_recall"] >= threshold, "TFR(recall) below threshold!"

    # Optional: quick 2D slice plot (y,theta) with velocities 0
    n = 220
    y = np.linspace(CartPole.x_low[0], CartPole.x_high[0], n)
    th = np.linspace(CartPole.x_low[1], CartPole.x_high[1], n)
    Y, TH = np.meshgrid(y, th)

    X = np.zeros((4, n*n), dtype=np.float32)
    X[0,:] = Y.flatten()
    X[1,:] = TH.flatten()
    X[2,:] = 0.0
    X[3,:] = 0.0

    with torch.no_grad():
        Vp = V_pre(torch.from_numpy(X.T)).cpu().numpy().reshape(n,n)
        Vf = V_fine(torch.from_numpy(X.T)).cpu().numpy().reshape(n,n)

    plt.figure(figsize=(10,8))
    plt.contour(Y, TH, Vp, levels=[0], colors="orange", linewidths=2)
    plt.contour(Y, TH, Vf, levels=[0], colors="blue", linewidths=2.5)
    plt.xlabel("y"); plt.ylabel("theta")
    plt.title("CartPole slice: V=0 pretrain (orange) vs finetune (blue)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = "plots/cart_pole_pre_vs_fine_slice.png"
    plt.savefig(out, dpi=300)
    print("Saved:", out)

if __name__ == "__main__":
    main()