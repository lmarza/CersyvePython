"""Fine-tune value function using counterexample-guided search."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import Optional
import os

from .buffer import Buffer
from .search import boundary_guided_search, projected_boundary_search, filter_counterexample
from .milp import milp_verify
from .enumeration import (
    BinaryDecisionWrapperCBF,
    enumerate_unsafe_regions,
    sample_from_regions
)
from .utils import uniform, set_seed
from .model import create_value_constraint_model, create_value_next_value_model


def finetune_value(
    V_model: nn.Module,
    f_pi_model: nn.Module,
    h_model: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    lr: float = 1e-4,
    max_iter: int = 100000,
    search_size: int = 1000,
    bnd_ratio: float = 0.1,
    bnd_ratio_avg: float = 0.9,
    min_bnd_ratio: float = 0.01,
    max_bnd_ratio: float = 1.0,
    bnd_eps: float = 0.1,
    search_method: str = "BGB",  # "BGB" / "PGD-B" / "PBS"
    pgd_step: int = 10,
    pgd_eps: float = 0.1,
    backtrack_step: int = 20,
    length_discount: float = 0.8,
    direct_discount: float = 0.5,
    tol: float = 1e-4,
    capacity: int = 10000,
    sample_size: int = 100,
    search_stop: int = 1000,
    replay: int = 1,
    max_skip: int = 1000,
    reg_method: Optional[str] = "ESR",  # "ESR" / "RSR" / None
    esr_max_con: Optional[int] = None,
    eps_h: float = 0.01,
    eps_v: float = 0.01,
    reg_coef: float = 0.1,
    log_dir: Optional[str] = None,
    eval_every: int = 10,
    save_every: int = 1000,
    device: str = 'cpu',
    seed: int = 42
) -> str:
    """
    Fine-tune value function using counterexample-guided search.
    
    The algorithm iteratively:
    1. Searches for counterexamples near the boundary V(x) = 0
    2. Trains the value function to exclude counterexamples
    3. Uses regularization to encourage conservative estimates
    
    Args:
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        h_model: Constraint model
        x_low, x_high: State space bounds
        lr: Learning rate
        max_iter: Maximum number of iterations
        search_size: Number of samples for boundary search
        bnd_ratio: Initial ratio of boundary samples
        bnd_ratio_avg: Exponential moving average factor for boundary ratio
        min_bnd_ratio, max_bnd_ratio: Bounds on boundary ratio
        bnd_eps: Tolerance for boundary sampling
        search_method: Search algorithm ("BGB", "PGD-B", or "PBS")
        pgd_step: Number of PGD steps
        pgd_eps: PGD step size
        backtrack_step: Number of backtracking steps
        length_discount: Discount factor for step length
        direct_discount: Discount factor for boundary guidance
        tol: Tolerance for counterexample detection
        capacity: Buffer capacity
        sample_size: Number of samples to train on per iteration
        search_stop: Stop searching after this many counterexamples found
        replay: Number of times to replay each counterexample
        max_skip: Maximum iterations without finding counterexamples before stopping
        reg_method: Regularization method ("ESR", "RSR", or None)
        esr_max_con: Maximum counterexamples for ESR
        eps_h: Constraint threshold for ESR
        eps_v: Value threshold for ESR
        reg_coef: Regularization coefficient
        log_dir: Directory for logs
        eval_every: Evaluate feasible rate every N iterations
        save_every: Save model every N iterations
        device: Device to train on
        seed: Random seed for reproducibility
        
    Returns:
        log_path: Path to saved model and logs
    """
    set_seed(seed)
    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    h_model = h_model.to(device)
    
    skipped = 0
    verified = 0
    
    buffer = Buffer(capacity, len(x_low))
    optimizer = torch.optim.Adam(V_model.parameters(), lr=lr)
    
    if log_dir is None:
        log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_path)
    
    for i in tqdm(range(max_iter), desc="Finetuning value function"):
        # Search for counterexamples
        if len(buffer) < search_stop:
            # Sample boundary states
            x = uniform(x_low, x_high, round(search_size / bnd_ratio))
            x_t = torch.from_numpy(x).to(device)
            with torch.no_grad():
                v = V_model(x_t.T).T.cpu().numpy()[0, :]
            
            x_bnd = x[:, (v > -bnd_eps) & (v <= tol)]
            bnd_ratio = bnd_ratio_avg * bnd_ratio + (1 - bnd_ratio_avg) * x_bnd.shape[1] / x.shape[1]
            bnd_ratio = np.clip(bnd_ratio, min_bnd_ratio, max_bnd_ratio)
            
            # Run adversarial search
            if search_method == "BGB":
                x_pgd = boundary_guided_search(
                    x_bnd, x_low, x_high, h_model, V_model, f_pi_model,
                    pgd_step=pgd_step, pgd_eps=pgd_eps, backtrack_step=backtrack_step,
                    length_discount=length_discount, bound_guide=True,
                    direct_discount=direct_discount, tol=tol, device=device
                )
            elif search_method == "PGD-B":
                x_pgd = boundary_guided_search(
                    x_bnd, x_low, x_high, h_model, V_model, f_pi_model,
                    pgd_step=pgd_step, pgd_eps=pgd_eps, backtrack_step=backtrack_step,
                    length_discount=length_discount, bound_guide=False, tol=tol, device=device
                )
            elif search_method == "PBS":
                x_pgd = projected_boundary_search(
                    x_bnd, x_low, x_high, h_model, V_model, f_pi_model,
                    tol=tol, device=device
                )
            
            # Filter counterexamples
            con, inv = filter_counterexample(x_pgd, h_model, V_model, f_pi_model, tol=tol, device=device)
            ce = con | inv
            if np.any(ce):
                buffer.push(x_pgd[:, ce])
            
            writer.add_scalar('finetune/searched_boundary_states', x_bnd.shape[1], i)
            writer.add_scalar('finetune/boundary_state_ratio', bnd_ratio, i)
            writer.add_scalar('finetune/searched_constraint_counterexample', con.sum(), i)
            writer.add_scalar('finetune/searched_invariance_counterexample', inv.sum(), i)
        
        # Train on counterexamples
        if len(buffer) > 0:
            skipped = 0
            
            n = min(sample_size, len(buffer))
            x, c = buffer.pop(n)
            
            # Re-filter counterexamples (some may no longer be violations)
            con, inv = filter_counterexample(x, h_model, V_model, f_pi_model, tol=tol, device=device)
            x_con, x_inv = x[:, con], x[:, inv]
            
            # Update replay counts
            c[con | inv] = 0
            c[~con & ~inv] += 1
            push_idx = c < replay
            if np.any(push_idx):
                buffer.push(x[:, push_idx], c[push_idx])
            
            n_con, n_inv = x_con.shape[1], x_inv.shape[1]
            
            # Sample regularization states
            if reg_method is not None and (n_con + n_inv > 0) and \
               (esr_max_con is None or (n_con + n_inv < esr_max_con)):
                if reg_method == "ESR":
                    # Entering State Regularization
                    x_reg = uniform(x_low, x_high, search_size)
                    x_reg_t = torch.from_numpy(x_reg).to(device)
                    with torch.no_grad():
                        h_reg = h_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        v_reg = V_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        x_reg_prime = f_pi_model(x_reg_t)
                        v_reg_prime = V_model(x_reg_prime.T).T.cpu().numpy()[0, :]
                    
                    # Entering states: safe, positive value, next state has negative value
                    entering = (h_reg <= -eps_h) & (v_reg > 0) & \
                               (v_reg <= eps_v) & (v_reg_prime <= -eps_v)
                    x_reg = x_reg[:, entering]
                    n_reg = x_reg.shape[1]
                elif reg_method == "RSR":
                    # Random State Regularization
                    x_reg = uniform(x_low, x_high, search_size)
                    n_reg = x_reg.shape[1]
            else:
                n_reg = 0
            
            # Compute loss and update
            optimizer.zero_grad()
            
            loss = 0.0
            if n_con > 0:
                x_con_t = torch.from_numpy(x_con).to(device)
                con_loss = -V_model(x_con_t.T).T.sum()
                loss += con_loss
            
            if n_inv > 0:
                x_inv_t = torch.from_numpy(x_inv).to(device)
                v_inv = V_model(x_inv_t.T).T
                x_inv_prime = f_pi_model(x_inv_t)
                v_inv_prime = V_model(x_inv_prime.T).T
                inv_loss = (-v_inv + v_inv_prime).sum()
                loss += inv_loss
            
            if n_con + n_inv > 0:
                loss = loss / (n_con + n_inv)
            
            if n_reg > 0:
                x_reg_t = torch.from_numpy(x_reg).to(device)
                reg_loss = V_model(x_reg_t.T).T.mean()
                loss = loss + reg_coef * reg_loss
            
            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()
            
            writer.add_scalar('finetune/sample_size', n, i)
            writer.add_scalar('finetune/value_loss', loss.item() if isinstance(loss, torch.Tensor) else 0.0, i)
            writer.add_scalar('finetune/sampled_constraint_counterexample', n_con, i)
            writer.add_scalar('finetune/sampled_invariance_counterexample', n_inv, i)
            if reg_method is not None:
                writer.add_scalar('finetune/regularization_state', n_reg, i)
        else:
            skipped += 1
        
        writer.add_scalar('finetune/total_counterexample', len(buffer), i)
        writer.add_scalar('finetune/skipped_update', skipped, i)
        writer.add_scalar('finetune/verified_times', verified, i)
        
        # Evaluation
        if i % eval_every == 0:
            x_eval = uniform(x_low, x_high, search_size)
            x_eval_t = torch.from_numpy(x_eval).to(device)
            with torch.no_grad():
                fea_rate = (V_model(x_eval_t.T).T <= 0).float().mean().item()
            writer.add_scalar('finetune/predicted_feasible_rate', fea_rate, i)
        
        # Save checkpoint
        if (i + 1) % save_every == 0:
            torch.save(V_model.state_dict(), os.path.join(log_path, f'V_finetune_iter{i+1}.pth'))
        
        # Check stopping criterion
        if skipped >= max_skip:
            print(f"\nStopping: no counterexamples found for {max_skip} iterations")
            break
    
    # Save final model
    torch.save(V_model.state_dict(), os.path.join("../../models/double_integrator/", 'V_finetune.pth'))
    writer.close()
    
    return log_path


def finetune_value_w_MILP(
    V_model: nn.Module,
    f_pi_model: nn.Module,
    h_model: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    lr: float = 1e-4,
    max_iter: int = 100000,
    search_size: int = 1000,
    bnd_ratio: float = 0.1,
    bnd_ratio_avg: float = 0.9,
    min_bnd_ratio: float = 0.01,
    max_bnd_ratio: float = 1.0,
    bnd_eps: float = 0.1,
    search_method: str = "BGB",
    pgd_step: int = 10,
    pgd_eps: float = 0.1,
    backtrack_step: int = 20,
    length_discount: float = 0.8,
    direct_discount: float = 0.5,
    tol: float = 1e-4,
    capacity: int = 10000,
    sample_size: int = 100,
    search_stop: int = 1000,
    replay: int = 1,
    max_skip: int = 1000,
    reg_method: Optional[str] = "ESR",
    esr_max_con: Optional[int] = None,
    eps_h: float = 0.01,
    eps_v: float = 0.01,
    reg_coef: float = 0.1,
    milp_timeout: int = 300,
    log_dir: Optional[str] = None,
    eval_every: int = 10,
    save_every: int = 1000,
    device: str = 'cpu',
    seed: int = 42
) -> str:
    """
    Fine-tune value function with MILP formal verification.

    Identical to ``finetune_value`` but when the sampling-based search
    finds 0 violations the algorithm runs a MILP check:
      - If MILP certifies correctness -> training stops (formally verified).
      - If MILP finds a counterexample -> it is pushed into the buffer and
        training continues.
      - If MILP is inconclusive (timeout) -> training continues.

    Args:
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        h_model: Constraint model
        x_low, x_high: State space bounds
        lr: Learning rate
        max_iter: Maximum number of iterations
        search_size: Number of samples for boundary search
        bnd_ratio: Initial ratio of boundary samples
        bnd_ratio_avg: EMA factor for boundary ratio
        min_bnd_ratio, max_bnd_ratio: Bounds on boundary ratio
        bnd_eps: Tolerance for boundary sampling
        search_method: Search algorithm ("BGB", "PGD-B", or "PBS")
        pgd_step: Number of PGD steps
        pgd_eps: PGD step size
        backtrack_step: Number of backtracking steps
        length_discount: Discount factor for step length
        direct_discount: Discount factor for boundary guidance
        tol: Tolerance for counterexample detection
        capacity: Buffer capacity
        sample_size: Number of samples to train on per iteration
        search_stop: Stop searching after this many counterexamples found
        replay: Number of times to replay each counterexample
        max_skip: Iterations without counterexamples before triggering MILP
        reg_method: Regularization method ("ESR", "RSR", or None)
        esr_max_con: Maximum counterexamples for ESR
        eps_h: Constraint threshold for ESR
        eps_v: Value threshold for ESR
        reg_coef: Regularization coefficient
        milp_timeout: Time limit (seconds) per MILP sub-problem
        log_dir: Directory for logs
        eval_every: Evaluate feasible rate every N iterations
        save_every: Save model every N iterations
        device: Device to train on
        seed: Random seed for reproducibility

    Returns:
        log_path: Path to saved model and logs
    """
    set_seed(seed)
    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    h_model = h_model.to(device)

    skipped = 0
    verified = 0
    formally_verified = False

    buffer = Buffer(capacity, len(x_low))
    optimizer = torch.optim.Adam(V_model.parameters(), lr=lr)

    if log_dir is None:
        log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"finetune_milp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_path)

    for i in tqdm(range(max_iter), desc="Finetuning value function (MILP)"):
        # ── Search for counterexamples ──
        if len(buffer) < search_stop:
            x = uniform(x_low, x_high, round(search_size / bnd_ratio))
            x_t = torch.from_numpy(x).to(device)
            with torch.no_grad():
                v = V_model(x_t.T).T.cpu().numpy()[0, :]

            x_bnd = x[:, (v > -bnd_eps) & (v <= tol)]
            bnd_ratio = bnd_ratio_avg * bnd_ratio + (1 - bnd_ratio_avg) * x_bnd.shape[1] / x.shape[1]
            bnd_ratio = np.clip(bnd_ratio, min_bnd_ratio, max_bnd_ratio)

            if search_method == "BGB":
                x_pgd = boundary_guided_search(
                    x_bnd, x_low, x_high, h_model, V_model, f_pi_model,
                    pgd_step=pgd_step, pgd_eps=pgd_eps, backtrack_step=backtrack_step,
                    length_discount=length_discount, bound_guide=True,
                    direct_discount=direct_discount, tol=tol, device=device
                )
            elif search_method == "PGD-B":
                x_pgd = boundary_guided_search(
                    x_bnd, x_low, x_high, h_model, V_model, f_pi_model,
                    pgd_step=pgd_step, pgd_eps=pgd_eps, backtrack_step=backtrack_step,
                    length_discount=length_discount, bound_guide=False, tol=tol, device=device
                )
            elif search_method == "PBS":
                x_pgd = projected_boundary_search(
                    x_bnd, x_low, x_high, h_model, V_model, f_pi_model,
                    tol=tol, device=device
                )

            con, inv = filter_counterexample(x_pgd, h_model, V_model, f_pi_model, tol=tol, device=device)
            ce = con | inv
            if np.any(ce):
                buffer.push(x_pgd[:, ce])

            writer.add_scalar('finetune/searched_boundary_states', x_bnd.shape[1], i)
            writer.add_scalar('finetune/boundary_state_ratio', bnd_ratio, i)
            writer.add_scalar('finetune/searched_constraint_counterexample', con.sum(), i)
            writer.add_scalar('finetune/searched_invariance_counterexample', inv.sum(), i)

        # ── Train on counterexamples ──
        if len(buffer) > 0:
            skipped = 0

            n = min(sample_size, len(buffer))
            x, c = buffer.pop(n)

            con, inv = filter_counterexample(x, h_model, V_model, f_pi_model, tol=tol, device=device)
            x_con, x_inv = x[:, con], x[:, inv]

            c[con | inv] = 0
            c[~con & ~inv] += 1
            push_idx = c < replay
            if np.any(push_idx):
                buffer.push(x[:, push_idx], c[push_idx])

            n_con, n_inv = x_con.shape[1], x_inv.shape[1]

            # Regularization
            if reg_method is not None and (n_con + n_inv > 0) and \
               (esr_max_con is None or (n_con + n_inv < esr_max_con)):
                if reg_method == "ESR":
                    x_reg = uniform(x_low, x_high, search_size)
                    x_reg_t = torch.from_numpy(x_reg).to(device)
                    with torch.no_grad():
                        h_reg = h_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        v_reg = V_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        x_reg_prime = f_pi_model(x_reg_t)
                        v_reg_prime = V_model(x_reg_prime.T).T.cpu().numpy()[0, :]
                    entering = (h_reg <= -eps_h) & (v_reg > 0) & \
                               (v_reg <= eps_v) & (v_reg_prime <= -eps_v)
                    x_reg = x_reg[:, entering]
                    n_reg = x_reg.shape[1]
                elif reg_method == "RSR":
                    x_reg = uniform(x_low, x_high, search_size)
                    n_reg = x_reg.shape[1]
            else:
                n_reg = 0

            # Gradient step
            optimizer.zero_grad()
            loss = 0.0

            if n_con > 0:
                x_con_t = torch.from_numpy(x_con).to(device)
                con_loss = -V_model(x_con_t.T).T.sum()
                loss += con_loss

            if n_inv > 0:
                x_inv_t = torch.from_numpy(x_inv).to(device)
                v_inv = V_model(x_inv_t.T).T
                x_inv_prime = f_pi_model(x_inv_t)
                v_inv_prime = V_model(x_inv_prime.T).T
                inv_loss = (-v_inv + v_inv_prime).sum()
                loss += inv_loss

            if n_con + n_inv > 0:
                loss = loss / (n_con + n_inv)

            if n_reg > 0:
                x_reg_t = torch.from_numpy(x_reg).to(device)
                reg_loss = V_model(x_reg_t.T).T.mean()
                loss = loss + reg_coef * reg_loss

            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()

            writer.add_scalar('finetune/sample_size', n, i)
            writer.add_scalar('finetune/value_loss', loss.item() if isinstance(loss, torch.Tensor) else 0.0, i)
            writer.add_scalar('finetune/sampled_constraint_counterexample', n_con, i)
            writer.add_scalar('finetune/sampled_invariance_counterexample', n_inv, i)
            if reg_method is not None:
                writer.add_scalar('finetune/regularization_state', n_reg, i)
        else:
            skipped += 1

        writer.add_scalar('finetune/total_counterexample', len(buffer), i)
        writer.add_scalar('finetune/skipped_update', skipped, i)
        writer.add_scalar('finetune/verified_times', verified, i)

        # ── Evaluation ──
        if i % eval_every == 0:
            x_eval = uniform(x_low, x_high, search_size)
            x_eval_t = torch.from_numpy(x_eval).to(device)
            with torch.no_grad():
                fea_rate = (V_model(x_eval_t.T).T <= 0).float().mean().item()
            writer.add_scalar('finetune/predicted_feasible_rate', fea_rate, i)

        # Save checkpoint
        if (i + 1) % save_every == 0:
            torch.save(V_model.state_dict(), os.path.join(log_path, f'V_finetune_milp_iter{i+1}.pth'))

        # ── MILP verification when no counterexamples found ──
        if skipped >= max_skip:
            print(f"\n  No counterexamples for {max_skip} iterations — running MILP verification...")
            milp_result, milp_ce = milp_verify(
                V_model, h_model, f_pi_model, x_low, x_high,
                timeout=milp_timeout, verbose=True
            )
            if milp_result is True:
                print("  FORMALLY VERIFIED: no violations exist (MILP certificate).")
                formally_verified = True
                writer.add_scalar('finetune/milp_verified', 1, i)
                break
            elif milp_result is False:
                n_ce = milp_ce.shape[1]
                print(f"  MILP found {n_ce} counterexample(s) — pushing to buffer.")
                n_push = min(n_ce, len(buffer.vacant))
                if n_push > 0:
                    buffer.push(milp_ce[:, :n_push])
                skipped = 0  # reset — we have new work to do
                writer.add_scalar('finetune/milp_counterexamples', n_ce, i)
            else:
                print("  MILP inconclusive (timeout) — continuing training.")
                skipped = 0  # reset and keep searching
                writer.add_scalar('finetune/milp_inconclusive', 1, i)

    # Save final model
    torch.save(V_model.state_dict(), os.path.join("../../models/double_integrator/", 'V_finetune_milp.pth'))
    writer.close()

    if formally_verified:
        print(f"\nTraining complete — model FORMALLY VERIFIED. Saved to models/double_integrator/")
    else:
        print(f"\nTraining complete (max_iter reached). Model saved to models/double_integrator/")

    return log_path


def finetune_value_enum_MILP(
    V_model: nn.Module,
    f_pi_model: nn.Module,
    h_model: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    lr: float = 1e-4,
    max_iter: int = 100000,
    tol: float = 1e-4,
    capacity: int = 10000,
    sample_size: int = 100,
    replay: int = 1,
    max_skip: int = 1000,
    # ── Enumeration parameters ──
    enum_freq: int = 1000,
    enum_n_samples: int = 20000,
    enum_coverage: float = 70.0,
    enum_n_trees: int = 2000,
    enum_max_depth: int = 20,
    enum_n_train: int = 40000,
    refill_threshold: int = 500,
    # ── Neighborhood fallback ──
    fallback_eval_rounds: int = 5,
    fallback_eval_samples: int = 100000,
    fallback_n_neighbors: int = 5000,
    fallback_radius: float = 0.05,
    # ── Regularization ──
    search_size: int = 1000,
    reg_method: Optional[str] = "ESR",
    esr_max_con: Optional[int] = None,
    eps_h: float = 0.01,
    eps_v: float = 0.01,
    reg_coef: float = 0.1,
    # ── MILP ──
    milp_timeout: int = 300,
    # ── Logging / misc ──
    log_dir: Optional[str] = None,
    eval_every: int = 100,
    save_every: int = 5000,
    save_path: Optional[str] = None,
    device: str = 'cpu',
    seed: int = 42
) -> str:
    """
    Fine-tune value function using **enumeration-based search + MILP verification**.

    Instead of PGD adversarial attacks this function uses Random-Forest-based
    enumeration to discover *all* unsafe hyperrectangles
    (regions where ``V(x) <= 0`` but ``h(x) > 0`` or ``V(f_pi(x)) > 0``).
    Counterexamples are then sampled volume-proportionally from those regions
    and pushed into a replay buffer.  The gradient step, ESR/RSR
    regularisation and buffer logic are **identical** to ``finetune_value``.

    **Why this is faster / more robust than PGD-based search:**

    1. *Global coverage* – enumeration finds the full landscape of unsafe
       regions rather than relying on sparse local PGD trajectories.
    2. *Volume-aware sampling* – larger violation regions receive more
       gradient signal, shrinking them efficiently.
    3. *Amortised cost* – regions are re-used for many cheap refills between
       expensive re-enumerations.
    4. *Neighbourhood amplification* – when regions go stale a local
       densification pass finds residual boundary violations that would take
       many PGD rounds to discover.
    5. *Faster MILP convergence* – the global view means fewer training
       iterations before the MILP can certify correctness.

    When sampling finds no more violations for ``max_skip`` iterations the
    algorithm switches to MILP formal verification:

    * **Certified** → training stops.
    * **Counterexample found** → pushed to buffer; training continues.
    * **Inconclusive (timeout)** → training continues.

    Args:
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        h_model: Constraint model
        x_low, x_high: State-space bounds (1-D arrays)
        lr: Learning rate
        max_iter: Maximum training iterations
        tol: Tolerance for counterexample detection
        capacity: Replay-buffer capacity
        sample_size: Counterexamples drawn from buffer per gradient step
        replay: Maximum replay count before discarding a sample
        max_skip: Empty iterations before triggering MILP
        enum_freq: Re-enumerate every N iterations
        enum_n_samples: Samples drawn from regions per refill
        enum_coverage: Target RF coverage (%)
        enum_n_trees: Number of RF trees
        enum_max_depth: Maximum RF tree depth
        enum_n_train: RF training-set size
        refill_threshold: Refill from regions when buffer drops below this
        fallback_eval_rounds: Uniform-sampling rounds for neighbourhood fallback
        fallback_eval_samples: Samples per fallback evaluation round
        fallback_n_neighbors: Total neighbourhood samples per fallback
        fallback_radius: Neighbourhood hypercube half-width
        search_size: Sample size for ESR evaluation
        reg_method: Regularisation method ("ESR", "RSR", or None)
        esr_max_con: Maximum counterexamples for ESR
        eps_h: Constraint threshold for ESR
        eps_v: Value threshold for ESR
        reg_coef: Regularisation coefficient
        milp_timeout: Time limit (s) per MILP sub-problem
        log_dir: Directory for TensorBoard logs / checkpoints
        eval_every: Log feasible-rate every N iterations
        save_every: Save a checkpoint every N iterations
        save_path: Final model save path (default: ``log_dir/../V_finetune_enum_milp.pth``)
        device: Torch device
        seed: Random seed

    Returns:
        log_path: Directory containing logs and checkpoints
    """
    set_seed(seed)
    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    h_model = h_model.to(device)

    x_dim = len(x_low)
    input_domain = np.stack([x_low, x_high], axis=1)  # (x_dim, 2)

    skipped = 0
    formally_verified = False
    unsafe_regions: list = []
    enum_stale = False          # regions no longer produce violations
    milp_calls = 0              # adaptive skip shrinks after each MILP call
    current_max_skip = max_skip

    buffer = Buffer(capacity, x_dim)
    optimizer = torch.optim.Adam(V_model.parameters(), lr=lr)

    if log_dir is None:
        log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"finetune_enum_milp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    writer = SummaryWriter(log_path)

    # ================================================================
    #  Helper: neighbourhood amplification around seed points
    # ================================================================
    def _amplify_seeds(seeds: np.ndarray) -> int:
        """Sample dense neighbourhoods around *seeds* (x_dim, N),
        filter for true violations, push to buffer.

        Returns number of violations pushed.
        """
        n_seeds = seeds.shape[1]
        if n_seeds == 0:
            return 0

        n_per_seed = max(1, fallback_n_neighbors // n_seeds)
        neighbours: list = []
        for j in range(n_seeds):
            centre = seeds[:, j:j + 1]
            noise = np.random.uniform(
                -fallback_radius, fallback_radius,
                size=(x_dim, n_per_seed)
            ).astype(np.float32)
            pts = np.clip(centre + noise, x_low[:, None], x_high[:, None])
            neighbours.append(pts)

        X_nbr = np.concatenate(neighbours, axis=1)
        con, inv = filter_counterexample(
            X_nbr, h_model, V_model, f_pi_model, tol=tol, device=device
        )
        ce_nbr = con | inv

        X_all = np.concatenate([seeds, X_nbr[:, ce_nbr]], axis=1)
        n_push = min(X_all.shape[1], len(buffer.vacant))
        if n_push > 0:
            buffer.push(X_all[:, :n_push])
        return n_push

    def _neighbourhood_fallback() -> int:
        """Large-scale uniform search + neighbourhood densification.

        Returns the number of violations pushed to the buffer.
        """
        all_violations = []
        for _ in range(fallback_eval_rounds):
            X_fb = uniform(x_low, x_high, fallback_eval_samples)
            con, inv = filter_counterexample(
                X_fb, h_model, V_model, f_pi_model, tol=tol, device=device
            )
            ce = con | inv
            if ce.sum() > 0:
                all_violations.append(X_fb[:, ce])

        if len(all_violations) == 0:
            return 0

        seeds = np.concatenate(all_violations, axis=1)
        return _amplify_seeds(seeds)

    # ================================================================
    #  Main training loop
    # ================================================================
    for i in tqdm(range(max_iter), desc="Finetuning (enum+MILP)"):

        # ── 1. Enumeration search (periodic) ──────────────────────
        #   Run at iter 0, then every enum_freq iterations.
        #   Skip only if regions are still productive (not stale).
        need_enum = (i == 0) or (i % enum_freq == 0)

        if need_enum:
            wrapper = BinaryDecisionWrapperCBF(
                V_model, h_model, f_pi_model, tol=tol
            )
            unsafe_regions, stats = enumerate_unsafe_regions(
                wrapper_model=wrapper,
                input_domain=input_domain,
                n_training_samples=enum_n_train,
                n_decision_trees=enum_n_trees,
                max_depth=enum_max_depth,
                enumerate_safe_regions=False,
                incremental_coverage=True,
                wilks_filtering=False,
                wilks_samples=10,
                sample_size_evaluation=enum_n_train,
                desired_coverage=enum_coverage,
                digits_to_consider=0,
                seed=seed,
                verbose=False,
                device=device
            )
            enum_stale = (len(unsafe_regions) == 0)
            writer.add_scalar('enum/n_regions', len(unsafe_regions), i)
            writer.add_scalar('enum/coverage', stats.get('coverage', 0.0), i)
            if not enum_stale:
                tqdm.write(f"  [iter {i}] Enumeration: {len(unsafe_regions)} unsafe "
                           f"regions, coverage={stats.get('coverage', 0):.1f}%")
            else:
                tqdm.write(f"  [iter {i}] Enumeration: no unsafe regions found.")

        # ── 2. Refill buffer from known regions (cheap) ───────────
        if len(buffer) < refill_threshold and len(unsafe_regions) > 0:
            X_cand = sample_from_regions(
                unsafe_regions, n_samples=enum_n_samples, digits_to_consider=0
            ).T.astype(np.float32)

            con, inv = filter_counterexample(
                X_cand, h_model, V_model, f_pi_model, tol=tol, device=device
            )
            ce = con | inv
            n_found = int(ce.sum())

            if n_found > 0:
                n_push = min(n_found, len(buffer.vacant))
                if n_push > 0:
                    buffer.push(X_cand[:, ce][:, :n_push])
            else:
                # Regions no longer productive
                enum_stale = True
                unsafe_regions = []

            writer.add_scalar('enum/refill_found', n_found, i)

        # ── 3. Neighbourhood fallback (buffer empty, any time) ────
        #   No longer gated on enum_stale — always try when buffer
        #   is empty to avoid dead-zone iterations.
        if len(buffer) == 0:
            n_added = _neighbourhood_fallback()
            if n_added > 0:
                tqdm.write(f"  [iter {i}] Neighbourhood fallback: "
                           f"{n_added} violations → buffer")

        # ── 4. Single gradient step from buffer ───────────────────
        if len(buffer) > 0:
            skipped = 0

            n = min(sample_size, len(buffer))
            x, c = buffer.pop(n)

            # Re-filter (some violations may have been fixed)
            con, inv = filter_counterexample(
                x, h_model, V_model, f_pi_model, tol=tol, device=device
            )
            x_con, x_inv = x[:, con], x[:, inv]

            c[con | inv] = 0
            c[~con & ~inv] += 1
            push_idx = c < replay
            if np.any(push_idx):
                n_back = min(int(push_idx.sum()), len(buffer.vacant))
                if n_back > 0:
                    buffer.push(x[:, push_idx][:, :n_back], c[push_idx][:n_back])

            n_con, n_inv = x_con.shape[1], x_inv.shape[1]

            # ── Regularisation (identical to finetune_value) ──
            n_reg = 0
            x_reg = None
            if reg_method is not None and (n_con + n_inv > 0) and \
               (esr_max_con is None or (n_con + n_inv < esr_max_con)):
                if reg_method == "ESR":
                    x_reg_cand = uniform(x_low, x_high, search_size)
                    x_reg_t = torch.from_numpy(x_reg_cand).to(device)
                    with torch.no_grad():
                        h_reg = h_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        v_reg = V_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        x_reg_prime = f_pi_model(x_reg_t)
                        v_reg_prime = V_model(x_reg_prime.T).T.cpu().numpy()[0, :]
                    entering = (h_reg <= -eps_h) & (v_reg > 0) & \
                               (v_reg <= eps_v) & (v_reg_prime <= -eps_v)
                    x_reg = x_reg_cand[:, entering]
                    n_reg = x_reg.shape[1]
                elif reg_method == "RSR":
                    x_reg = uniform(x_low, x_high, search_size)
                    n_reg = x_reg.shape[1]

            # ── Gradient step ──
            optimizer.zero_grad()
            loss = 0.0

            if n_con > 0:
                x_con_t = torch.from_numpy(x_con).to(device)
                con_loss = -V_model(x_con_t.T).T.sum()
                loss += con_loss

            if n_inv > 0:
                x_inv_t = torch.from_numpy(x_inv).to(device)
                v_inv = V_model(x_inv_t.T).T
                x_inv_prime = f_pi_model(x_inv_t)
                v_inv_prime = V_model(x_inv_prime.T).T
                inv_loss = (-v_inv + v_inv_prime).sum()
                loss += inv_loss

            if n_con + n_inv > 0:
                loss = loss / (n_con + n_inv)

            if n_reg > 0:
                x_reg_t = torch.from_numpy(x_reg).to(device)
                reg_loss = V_model(x_reg_t.T).T.mean()
                loss = loss + reg_coef * reg_loss

            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()

            writer.add_scalar('finetune/sample_size', n, i)
            writer.add_scalar('finetune/value_loss',
                              loss.item() if isinstance(loss, torch.Tensor) else 0.0, i)
            writer.add_scalar('finetune/sampled_constraint_counterexample', n_con, i)
            writer.add_scalar('finetune/sampled_invariance_counterexample', n_inv, i)
            if reg_method is not None:
                writer.add_scalar('finetune/regularization_state', n_reg, i)
        else:
            skipped += 1

        writer.add_scalar('finetune/total_counterexample', len(buffer), i)
        writer.add_scalar('finetune/skipped_update', skipped, i)

        # ── 5. Evaluation ─────────────────────────────────────────
        if i % eval_every == 0:
            x_eval = uniform(x_low, x_high, search_size)
            x_eval_t = torch.from_numpy(x_eval).to(device)
            with torch.no_grad():
                fea_rate = (V_model(x_eval_t.T).T <= 0).float().mean().item()
            writer.add_scalar('finetune/predicted_feasible_rate', fea_rate, i)

        # ── 6. Checkpoint ─────────────────────────────────────────
        if (i + 1) % save_every == 0:
            torch.save(
                V_model.state_dict(),
                os.path.join(log_path, f'V_finetune_enum_milp_iter{i+1}.pth')
            )

        # ── 7. MILP verification when search is exhausted ─────────
        #   Uses adaptive skip: halves after each MILP call so the
        #   tail phase (sparse violations) converges faster.
        if skipped >= current_max_skip:
            tqdm.write(f"\n  No counterexamples for {current_max_skip} iterations "
                       f"— running MILP verification...")
            milp_result, milp_ce = milp_verify(
                V_model, h_model, f_pi_model, x_low, x_high,
                timeout=milp_timeout, verbose=True
            )
            milp_calls += 1

            if milp_result is True:
                tqdm.write("  FORMALLY VERIFIED: no violations exist "
                           "(MILP certificate).")
                formally_verified = True
                writer.add_scalar('finetune/milp_verified', 1, i)
                break

            elif milp_result is False:
                # ── Amplify the MILP counterexample ──
                n_ce = milp_ce.shape[1]
                tqdm.write(f"  MILP found {n_ce} counterexample(s) "
                           f"— amplifying with neighbourhood search...")
                n_amplified = _amplify_seeds(milp_ce)
                tqdm.write(f"  Amplified to {n_amplified} violations in buffer.")

                skipped = 0
                # Shrink max_skip: at most half, but never below 50
                current_max_skip = max(50, current_max_skip // 2)
                writer.add_scalar('finetune/milp_counterexamples', n_ce, i)
                writer.add_scalar('finetune/current_max_skip', current_max_skip, i)

            else:
                tqdm.write("  MILP inconclusive (timeout) "
                           "— continuing training.")
                skipped = 0
                current_max_skip = max(50, current_max_skip // 2)
                writer.add_scalar('finetune/milp_inconclusive', 1, i)

    # ── Save final model ──────────────────────────────────────────
    if save_path is None:
        save_path = os.path.join(log_path, 'V_finetune_enum_milp.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(V_model.state_dict(), save_path)
    writer.close()

    if formally_verified:
        print(f"\nTraining complete — model FORMALLY VERIFIED. Saved to {save_path}")
    else:
        print(f"\nTraining complete (max_iter reached). Model saved to {save_path}")

    return log_path
