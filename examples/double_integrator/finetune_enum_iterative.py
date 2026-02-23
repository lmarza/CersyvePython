"""Iterative Enumeration-Guided Fine-tuning.

Uses the EXACT same training loop as the original finetune_value()
(single gradient step per iteration, buffer, ESR regularization),
but replaces PGD adversarial search with enumeration-based search.

Key insight: the original works because it does ONE gradient step per iteration,
not hundreds of epochs. This prevents catastrophic forgetting.
"""

import os
import sys

from visualize_comparison import visualize_comparison
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

from systems.double_integrator import DoubleIntegrator
from cersyve import create_mlp, create_closed_loop_dynamics_model, utils
from cersyve.utils import uniform
from cersyve.enumeration import enumerate_unsafe_regions, BinaryDecisionWrapperCBF
from cersyve.search import filter_counterexample
from cersyve.buffer import Buffer


def load_models(device='cpu'):
    """Load dynamics, constraint, and pretrained value function models."""
    data_path = '../../data/double_integrator_data.npz'
    data_npz = np.load(data_path)
    data = {key: data_npz[key] for key in data_npz.keys()}

    dynamics_hidden_sizes = [32, 32]
    f_model = create_mlp(DoubleIntegrator.x_dim + DoubleIntegrator.u_dim, DoubleIntegrator.x_dim, dynamics_hidden_sizes)
    f_model.load_state_dict(torch.load('../../models/double_integrator/f.pth', map_location=device, weights_only=True))

    pi_model = DoubleIntegrator.create_policy()
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, DoubleIntegrator.x_low, DoubleIntegrator.x_high, DoubleIntegrator.u_dim
    )

    constraint_hidden_sizes = [16]
    h_model = create_mlp(DoubleIntegrator.x_dim, 1, constraint_hidden_sizes)
    h_model.load_state_dict(torch.load('../../models/double_integrator/h.pth', map_location=device, weights_only=True))

    value_hidden_sizes = [32, 32]
    V_model = create_mlp(DoubleIntegrator.x_dim, 1, value_hidden_sizes)
    V_model.load_state_dict(torch.load('../../models/double_integrator/V_pretrain.pth', map_location=device, weights_only=True))
    V_model.eval()

    return V_model, h_model, f_pi_model


def run_enumeration(V_model, h_model, f_pi_model, target_coverage=70.0, device='cpu'):
    """Run enumeration to find unsafe regions."""
    input_domain = np.array([DoubleIntegrator.x_low, DoubleIntegrator.x_high]).T
    wrapper = BinaryDecisionWrapperCBF(V_model, h_model, f_pi_model, tol=0.0)

    unsafe_regions, stats = enumerate_unsafe_regions(
        wrapper_model=wrapper,
        input_domain=input_domain,
        n_training_samples=40_000,
        n_decision_trees=2000,
        max_depth=20,
        enumerate_safe_regions=False,
        incremental_coverage=True,
        wilks_filtering=False,
        wilks_samples=10,
        sample_size_evaluation=40_000,
        desired_coverage=target_coverage,
        digits_to_consider=0,
        seed=42,
        verbose=False,
        device=device
    )
    return unsafe_regions, stats


def sample_from_regions(unsafe_regions, n_samples=10_000):
    """Sample points uniformly from unsafe regions."""
    if len(unsafe_regions) == 0:
        return np.zeros((DoubleIntegrator.x_dim, 0), dtype=np.float32)

    n_dims = len(unsafe_regions[0][0])
    volumes = np.array([np.prod([b[1] - b[0] for b in bounds]) for bounds in unsafe_regions])
    volumes = volumes / volumes.sum()

    samples_per_region = (volumes * n_samples).astype(int)
    samples_per_region[-1] = n_samples - samples_per_region[:-1].sum()

    samples = []
    for region_idx, bounds in enumerate(unsafe_regions):
        n = samples_per_region[region_idx]
        if n == 0:
            continue
        region_samples = np.zeros((n_dims, n), dtype=np.float32)
        for dim in range(n_dims):
            low, high = bounds[dim]
            region_samples[dim, :] = np.random.uniform(low, high, n)
        samples.append(region_samples)

    if len(samples) == 0:
        return np.zeros((n_dims, 0), dtype=np.float32)
    return np.concatenate(samples, axis=1)


def compute_tfr(V_model, h_model, f_pi_model, n_samples=100_000, horizon=100, device='cpu'):
    """Compute the True Feasible Rate (TFR).

    TFR = (# states with V(x)<=0 AND truly feasible) / (# truly feasible states)

    A state is truly feasible if a 100-step trajectory starting from it
    has no constraint violation (h(x) <= 0 at every step).
    """
    V_model.eval()
    X = uniform(DoubleIntegrator.x_low, DoubleIntegrator.x_high, n_samples)
    X_t = torch.from_numpy(X).to(device)  # [x_dim, N]

    with torch.no_grad():
        V = V_model(X_t.T).T.cpu().numpy()[0, :]  # [N]

        # Forward simulate and check constraints along the trajectory
        truly_feasible = np.ones(n_samples, dtype=bool)
        X_current = X_t.clone()  # [x_dim, N]
        for step in range(horizon):
            h_current = h_model(X_current.T).T.cpu().numpy()[0, :]
            truly_feasible &= (h_current <= 0)
            X_current = f_pi_model(X_current)

    n_truly_feasible = truly_feasible.sum()
    identified = (V <= 0) & truly_feasible
    n_identified = identified.sum()

    tfr = n_identified / max(n_truly_feasible, 1)
    return {
        'tfr': tfr,
        'n_truly_feasible': n_truly_feasible,
        'n_identified': n_identified,
        'truly_feasible_rate': n_truly_feasible / n_samples,
    }


def evaluate_model(V_model, h_model, f_pi_model, n_samples=40_000, device='cpu'):
    """Evaluate the current model."""
    V_model.eval()
    X_eval = uniform(DoubleIntegrator.x_low, DoubleIntegrator.x_high, n_samples)
    X_eval_t = torch.from_numpy(X_eval).to(device)

    with torch.no_grad():
        V_eval = V_model(X_eval_t.T).T.cpu().numpy()[0, :]
        h_eval = h_model(X_eval_t.T).T.cpu().numpy()[0, :]
        X_next = f_pi_model(X_eval_t)
        V_next = V_model(X_next.T).T.cpu().numpy()[0, :]

    safe = V_eval <= 0
    con = safe & (h_eval > 0)
    inv = safe & (V_next > 0)
    total = con | inv

    return {
        'safe_size': safe.sum() / n_samples,
        'n_con': con.sum(), 'n_inv': inv.sum(), 'n_total': total.sum(),
        'pct_con': con.sum() / max(safe.sum(), 1),
        'pct_inv': inv.sum() / max(safe.sum(), 1),
        'pct_total': total.sum() / max(safe.sum(), 1),
    }


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # ── Hyperparameters (mirroring original finetune_value defaults) ──
    LR = 1e-5
    MAX_ITER = 15_000       # Total single-step iterations
    SAMPLE_SIZE = 100        # Counterexamples per gradient step (from buffer)
    BUFFER_CAPACITY = 5_000   # Small buffer → ~10 steps per cycle, prevents over-correction
    REPLAY = 1               # Max replay count before discarding
    TOL = 1e-4

    # Enumeration search parameters
    ENUM_COVERAGE = 70.0
    ENUM_SAMPLES = 5_000     # Candidates sampled from regions
    SEARCH_FREQ = 500        # Re-enumerate every N iterations
    SEARCH_STOP = 500        # Stop searching when buffer exceeds this

    # ESR regularization (exactly like original)
    REG_METHOD = "ESR"
    REG_COEF = 0.1
    EPS_H = 0.001
    EPS_V = 0.001
    REG_SIZE = 1000

    # Stopping
    MAX_SKIP = 1000          # Stop after this many empty iterations
    EVAL_EVERY = 100
    SAVE_EVERY = 5000

    utils.set_seed(42)

    x_low = DoubleIntegrator.x_low
    x_high = DoubleIntegrator.x_high


    V_model, h_model, f_pi_model = load_models(device=DEVICE)
    V_model.to(DEVICE); h_model.to(DEVICE); f_pi_model.to(DEVICE)

    buffer = Buffer(BUFFER_CAPACITY, DoubleIntegrator.x_dim)
    optimizer = torch.optim.Adam(V_model.parameters(), lr=LR)
    skipped = 0
    unsafe_regions = []
    best_violations = float('inf')
    best_state = None
    empty_enum_count = 0  # Consecutive enumerations finding nothing

    # Initial evaluation
    metrics = evaluate_model(V_model, h_model, f_pi_model, device=DEVICE)
    print(f"\nInitial: safe region={metrics['safe_size']*100:.2f}%  "
          f"con={metrics['n_con']} | inv={metrics['n_inv']} | total={metrics['n_total']}\n")

    start = time.time()
    # ── Main loop: single gradient step per iteration ──
    for i in (range(MAX_ITER)):

        # ── Search phase: find counterexamples via enumeration ──
        if len(buffer) < SEARCH_STOP and (i % SEARCH_FREQ == 0 or len(unsafe_regions) == 0):
            print("\nRunning enumeration to find unsafe regions...")
            unsafe_regions, stats = run_enumeration(
                V_model, h_model, f_pi_model,
                target_coverage=ENUM_COVERAGE, device=DEVICE
            )
  
            if len(unsafe_regions) == 0:
                empty_enum_count += 1
                print(f"  No unsafe regions found! ({empty_enum_count} consecutive)")
                if empty_enum_count >= 3:
                    print("\nConverged: enumeration found no unsafe regions 3 times in a row.")
                    break
            else:
                empty_enum_count = 0

        # Sample from regions and filter to verified counterexamples
        if len(buffer) < SEARCH_STOP and len(unsafe_regions) > 0:
            X_candidates = sample_from_regions(unsafe_regions, n_samples=ENUM_SAMPLES)
            con, inv = filter_counterexample(
                X_candidates, h_model, V_model, f_pi_model, tol=TOL, device=DEVICE
            )
            ce = con | inv
            n_found = ce.sum()
            if n_found > 0:
                # Only push up to remaining capacity
                n_to_push = min(n_found, len(buffer.vacant))
                if n_to_push > 0:
                    ce_points = X_candidates[:, ce][:, :n_to_push]
                    buffer.push(ce_points)

        # ── Train phase: SINGLE gradient step (exactly like original) ──
        if len(buffer) > 0:
            skipped = 0
            n = min(SAMPLE_SIZE, len(buffer))
            x, c = buffer.pop(n)

            # Re-filter (some may have been fixed by previous updates)
            con, inv = filter_counterexample(x, h_model, V_model, f_pi_model, tol=TOL, device=DEVICE)
            x_con, x_inv = x[:, con], x[:, inv]

            # Update replay counts and push back
            c[con | inv] = 0
            c[~con & ~inv] += 1
            push_idx = c < REPLAY
            if np.any(push_idx):
                n_push_back = min(push_idx.sum(), len(buffer.vacant))
                if n_push_back > 0:
                    buffer.push(x[:, push_idx][:, :n_push_back], c[push_idx][:n_push_back])

            n_con, n_inv = x_con.shape[1], x_inv.shape[1]

            # ESR regularization
            n_reg = 0
            x_reg = None
            if REG_METHOD == "ESR" and (n_con + n_inv) > 0:
                x_reg_candidates = uniform(x_low, x_high, REG_SIZE)
                x_reg_t = torch.from_numpy(x_reg_candidates).to(DEVICE)
                with torch.no_grad():
                    h_r = h_model(x_reg_t.T).T.cpu().numpy()[0, :]
                    v_r = V_model(x_reg_t.T).T.cpu().numpy()[0, :]
                    x_r_prime = f_pi_model(x_reg_t)
                    v_r_prime = V_model(x_r_prime.T).T.cpu().numpy()[0, :]
                entering = (h_r <= -EPS_H) & (v_r > 0) & (v_r <= EPS_V) & (v_r_prime <= -EPS_V)
                x_reg = x_reg_candidates[:, entering]
                n_reg = x_reg.shape[1]

            # ── Single gradient step ──
            optimizer.zero_grad()
            loss = 0.0

            if n_con > 0:
                x_con_t = torch.from_numpy(x_con).to(DEVICE)
                con_loss = -V_model(x_con_t.T).T.sum()
                loss += con_loss

            if n_inv > 0:
                x_inv_t = torch.from_numpy(x_inv).to(DEVICE)
                v_inv = V_model(x_inv_t.T).T
                x_inv_prime = f_pi_model(x_inv_t)
                v_inv_prime = V_model(x_inv_prime.T).T
                inv_loss = (-v_inv + v_inv_prime).sum()
                loss += inv_loss

            if n_con + n_inv > 0:
                loss = loss / (n_con + n_inv)

            if n_reg > 0:
                x_reg_t = torch.from_numpy(x_reg).to(DEVICE)
                reg_loss = V_model(x_reg_t.T).T.mean()
                loss = loss + REG_COEF * reg_loss

            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()
        else:
            skipped += 1


        # ── Evaluation & best-model checkpointing ──
        if i % EVAL_EVERY == 0 and i > 0:
            metrics = evaluate_model(V_model, h_model, f_pi_model, device=DEVICE)
            n_viol = metrics['n_total']
            tag = ''
            if n_viol < best_violations:
                best_violations = n_viol
                best_state = {k: v.clone() for k, v in V_model.state_dict().items()}
                tag = '  ★ best'
            print(f"[iter {i}] safe region={metrics['safe_size']*100:.2f}%  "
                  f"con={metrics['n_con']} | inv={metrics['n_inv']}  "
                  f"total={metrics['n_total']} | buffer={len(buffer)}{tag}")
            if n_viol == 0:
                print("\nConverged: 0 violations!")
                break

        # Stopping criterion
        if skipped >= MAX_SKIP:
            print(f"\nStopping: no counterexamples for {MAX_SKIP} iterations")
            break

    
    # Restore best model and save
    if best_state is not None:
        V_model.load_state_dict(best_state)
    metrics = evaluate_model(V_model, h_model, f_pi_model, device=DEVICE)
    print(f"\nBest model: safe region={metrics['safe_size']*100:.2f}%  "
          f"con={metrics['n_con']} | inv={metrics['n_inv']} | total={metrics['n_total']}")
    print("Total training time: {:.2f} minutes".format((time.time() - start) / 60))

    # Compute True Feasible Rate (TFR)
    print("\nComputing True Feasible Rate (100-step trajectory check)...")
    tfr_metrics = compute_tfr(V_model, h_model, f_pi_model, n_samples=100_000, horizon=100, device=DEVICE)
    print(f"TFR = {tfr_metrics['tfr']*100:.2f}%  "f"({tfr_metrics['n_identified']}/{tfr_metrics['n_truly_feasible']} truly feasible states identified)")

    final_path = '../../models/double_integrator/V_finetune_enum.pth'
    torch.save(V_model.state_dict(), final_path)
    print(f"Model saved to {final_path}")

    visualize_comparison(fine_tune_model_path=final_path, name_file='plots/comparison_visualization_enum.png')