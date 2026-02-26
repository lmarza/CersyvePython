"""Fine-tune value function using counterexample-guided search."""

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import Optional
import os

from systems import DoubleIntegrator
from .buffer import Buffer
from .search import boundary_guided_search, projected_boundary_search, filter_counterexample
from .enumeration import (
    BinaryDecisionWrapperCBF,
    enumerate_unsafe_regions,
    sample_from_regions
)
from .utils import uniform, set_seed
from .model import create_value_constraint_model, create_value_next_value_model
from .verify import verify_value as verify_value_abcrown, ClosedLoopDynamicsBatchFirst, ValueConstraintPropertyBF, ClosedLoopDynamicsBatchFirstWrapper, ValueNextValuePropertyBF


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
    max_skip: int = 200,
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
    f_pi_model: Module = f_pi_model.to(device)
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
    # Property networks
    f_pi_bf = ClosedLoopDynamicsBatchFirstWrapper(f_pi_model).to(device).eval()
    V_h_model = ValueConstraintPropertyBF(V_model, h_model).to(device).eval()
    V_V_prime_model = ValueNextValuePropertyBF(V_model, f_pi_bf).to(device).eval()
    
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

        if skipped == max_skip:
            print("\n----- Verification Starts (alpha-beta-CROWN) -----")
            con_res, inv_res = verify_value_abcrown(
                x_low, x_high,
                V_h_model,
                V_V_prime_model,
                strict_margin=1e-6,
                device=device,
                timeout_s=300.0,
            )
            print("----- Verification Ends -----\n")

            torch.set_grad_enabled(True)
            V_model.train()

            verified += 1

            # Log status strings
            writer.add_text("abcrown/constraint_status", str(con_res.status), i)
            writer.add_text("abcrown/invariance_status", str(inv_res.status), i)

            def _holds(status: str) -> bool:
                s = (status or "").lower()
                # abcrown typically uses "verified"/"safe" for proved UNSAT, and "unsafe-*" for counterexamples
                return s.startswith("verified") or s.startswith("safe") or s == "holds"

            if _holds(con_res.status) and _holds(inv_res.status):
                print("Both properties verified. Stopping.")
                break

            any_added = False

            # If violated, push verifier counterexamples into the buffer
            if str(con_res.status).lower().startswith("unsafe"):
                if con_res.counterexample is not None:
                    # convert abcrown ce to a single-column batch to match buffer element shape [dim, N]
                    ce = con_res.counterexample.astype(np.float32).reshape(-1, 1)
                    buffer.push(ce)
                    print("Constraint counterexample:", con_res.counterexample)
                    any_added = True
                else:
                    print("Constraint unsafe but no counterexample extracted from verifier result.")

            if str(inv_res.status).lower().startswith("unsafe"):
                if inv_res.counterexample is not None:
                    ce = inv_res.counterexample.astype(np.float32).reshape(-1, 1)
                    buffer.push(ce)
                    print("Invariance counterexample:", inv_res.counterexample)
                    any_added = True
                else:
                    print("Invariance unsafe but no counterexample extracted from verifier result.")

            # If verifier found something, resume training; otherwise avoid infinite loops.
            if any_added:
                skipped = 0
                with torch.no_grad():
                    xt = torch.tensor([[-1.0], [-1.0]], device=device)  # [2,1]
                    v = V_model(xt.T).item()
                    h = h_model(xt.T).item()
                print(f"[dbg] V(-1,-1)={v:.6f}, h(-1,-1)={h:.6f}")
            else:
                print("Verification did not produce usable counterexamples. Stopping.")
                break
    
    # Save final model
    torch.save(V_model.state_dict(), os.path.join("../../models/double_integrator/", 'V_finetune.pth'))
    writer.close()
    
    return log_path


def finetune_value_enumeration(
    V_model: nn.Module,
    f_pi_model: nn.Module,
    h_model: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    lr: float = 1e-4,
    max_iter: int = 100,
    n_training_samples: int = 10000,
    n_decision_trees: int = 100,
    max_tree_depth: int = 10,
    tree_heuristic: str = 'gini',
    digits_to_consider: int = 3,
    wilks_filtering: bool = True,
    wilks_samples: int = 3500,
    sample_size_evaluation: int = 100000,
    desired_coverage: float = 70.0,
    enumeration_freq: int = 10,
    sample_size: int = 100,
    tol: float = 0.0,
    reg_method: Optional[str] = "ESR",
    eps_h: float = 0.01,
    eps_v: float = 0.01,
    reg_coef: float = 0.1,
    reg_size: int = 1000,
    log_dir: Optional[str] = None,
    eval_every: int = 10,
    save_every: int = 1000,
    seed: int = 42,
    device: str = 'cpu'
) -> str:
    """
    Fine-tune value function using enumeration-based counterexample search.
    
    This approach uses Random Forest to enumerate regions where V(x) <= 0
    (predicted safe) but actual safety is violated (h(x) > 0 or V(f(x)) > 0).
    
    The algorithm iteratively:
    1. Enumerates unsafe regions using RF-ProVe approach
    2. Samples counterexamples from these regions
    3. Trains the value function to exclude counterexamples
    4. Uses regularization to encourage conservative estimates
    
    Args:
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        h_model: Constraint model
        x_low, x_high: State space bounds
        lr: Learning rate
        max_iter: Maximum number of fine-tuning iterations
        n_training_samples: Number of samples for training RF
        n_decision_trees: Number of trees in RF
        max_tree_depth: Maximum depth of each tree
        tree_heuristic: Split criterion ('gini' or 'entropy')
        digits_to_consider: Decimal places to round samples
        wilks_filtering: Whether to use Wilks filtering
        wilks_samples: Number of samples for Wilks test
        sample_size_evaluation: Number of samples for coverage evaluation
        desired_coverage: Target coverage of unsafe regions
        enumeration_freq: Re-enumerate regions every N iterations
        sample_size: Number of counterexamples to sample per iteration
        tol: Tolerance for safety checks
        reg_method: Regularization method ("ESR", "RSR", or None)
        eps_h: Constraint threshold for ESR
        eps_v: Value threshold for ESR
        reg_coef: Regularization coefficient
        reg_size: Number of regularization samples
        log_dir: Directory for logs
        eval_every: Evaluate feasible rate every N iterations
        save_every: Save model every N iterations
        seed: Random seed
        device: Device to train on
        
    Returns:
        log_path: Path to saved model and logs
    """
    set_seed(seed)
    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    h_model = h_model.to(device)
    
    optimizer = torch.optim.Adam(V_model.parameters(), lr=lr)
    
    if log_dir is None:
        log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"finetune_enum_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_path)
    
    # Create binary decision wrapper
    wrapper_model = BinaryDecisionWrapperCBF(V_model, h_model, f_pi_model, tol=tol)
    
    unsafe_regions = []
    stats = {}
    
    for i in tqdm(range(max_iter), desc="Finetuning with enumeration"):
        # Re-enumerate unsafe regions periodically
        if i % enumeration_freq == 0:
            print(f"\n[Iteration {i}] Enumerating unsafe regions...")
            
            unsafe_regions, stats = enumerate_unsafe_regions(
                wrapper_model=wrapper_model,
                input_domain=np.column_stack([x_low, x_high]),
                n_training_samples=n_training_samples,
                n_decision_trees=n_decision_trees,
                max_depth=max_tree_depth,
                heuristic=tree_heuristic,
                digits_to_consider=digits_to_consider,
                wilks_filtering=wilks_filtering,
                wilks_samples=wilks_samples,
                sample_size_evaluation=sample_size_evaluation,
                desired_coverage=desired_coverage,
                enumerate_safe_regions=False,  # Enumerate unsafe regions for counterexample search
                seed=seed + i,
                verbose=True,
                device=device
            )
            
            writer.add_scalar('enumeration/num_unsafe_regions', stats['num_regions'], i)
            writer.add_scalar('enumeration/coverage', stats['coverage'], i)
            writer.add_scalar('enumeration/error_rate', stats['error_rate'], i)
            if 'pgd_violations' in stats:
                writer.add_scalar('enumeration/pgd_violations', stats['pgd_violations'], i)
            
            if len(unsafe_regions) == 0:
                print(f"[Iteration {i}] No unsafe regions found! Model may be verified.")
                break
        
        # Sample counterexamples from unsafe regions
        if len(unsafe_regions) > 0:
            x_unsafe = sample_from_regions(
                unsafe_regions,
                n_samples=sample_size,
                digits_to_consider=digits_to_consider
            )
            
            if x_unsafe.shape[0] > 0:
                # Verify counterexamples are still unsafe
                x_unsafe_t = torch.from_numpy(x_unsafe.astype(np.float32)).to(device)
                
                with torch.no_grad():
                    v_unsafe = V_model(x_unsafe_t)
                    h_unsafe = h_model(x_unsafe_t)
                    x_next_unsafe = f_pi_model(x_unsafe_t.T).T
                    v_next_unsafe = V_model(x_next_unsafe)
                
                # Filter to actual counterexamples
                v_safe = (v_unsafe.squeeze() <= tol).cpu().numpy()
                h_violated = (h_unsafe.squeeze() > tol).cpu().numpy()
                v_next_violated = (v_next_unsafe.squeeze() > tol).cpu().numpy()
                
                con_mask = v_safe & h_violated
                inv_mask = v_safe & v_next_violated
                ce_mask = con_mask | inv_mask
                
                n_con = np.sum(con_mask)
                n_inv = np.sum(inv_mask)
                
                writer.add_scalar('finetune/sampled_counterexamples', np.sum(ce_mask), i)
                writer.add_scalar('finetune/constraint_counterexamples', n_con, i)
                writer.add_scalar('finetune/invariance_counterexamples', n_inv, i)
                
                # Compute loss
                optimizer.zero_grad()
                loss = 0.0
                
                if n_con > 0:
                    x_con = x_unsafe_t[con_mask]
                    con_loss = -V_model(x_con).mean()
                    loss += con_loss
                
                if n_inv > 0:
                    x_inv = x_unsafe_t[inv_mask]
                    v_inv = V_model(x_inv)
                    x_inv_next = f_pi_model(x_inv.T).T
                    v_inv_next = V_model(x_inv_next)
                    inv_loss = (-v_inv + v_inv_next).mean()
                    loss += inv_loss
                
                # Regularization
                if reg_method is not None and (n_con + n_inv) > 0:
                    x_reg = uniform(x_low, x_high, reg_size)
                    x_reg_t = torch.from_numpy(x_reg).to(device)
                    
                    if reg_method == "ESR":
                        with torch.no_grad():
                            h_reg = h_model(x_reg_t.T).T.cpu().numpy()[0, :]
                            v_reg = V_model(x_reg_t.T).T.cpu().numpy()[0, :]
                            x_reg_prime = f_pi_model(x_reg_t)
                            v_reg_prime = V_model(x_reg_prime.T).T.cpu().numpy()[0, :]
                        
                        entering = (h_reg <= -eps_h) & (v_reg > 0) & \
                                   (v_reg <= eps_v) & (v_reg_prime <= -eps_v)
                        x_reg = x_reg[:, entering]
                        
                        if x_reg.shape[1] > 0:
                            x_reg_t = torch.from_numpy(x_reg).to(device)
                            reg_loss = V_model(x_reg_t.T).T.mean()
                            loss = loss + reg_coef * reg_loss
                            writer.add_scalar('finetune/regularization_states', x_reg.shape[1], i)
                    elif reg_method == "RSR":
                        reg_loss = V_model(x_reg_t.T).T.mean()
                        loss = loss + reg_coef * reg_loss
                        writer.add_scalar('finetune/regularization_states', reg_size, i)
                
                # Update model
                if isinstance(loss, torch.Tensor) and loss.item() > 0:
                    loss.backward()
                    optimizer.step()
                    writer.add_scalar('finetune/value_loss', loss.item(), i)
                else:
                    writer.add_scalar('finetune/value_loss', 0.0, i)
        
        # Evaluation
        if i % eval_every == 0:
            x_eval = uniform(x_low, x_high, sample_size_evaluation)
            x_eval_t = torch.from_numpy(x_eval).to(device)
            with torch.no_grad():
                fea_rate = (V_model(x_eval_t.T).T <= 0).float().mean().item()
            writer.add_scalar('finetune/predicted_feasible_rate', fea_rate, i)
        
        # Save checkpoint
        if (i + 1) % save_every == 0:
            torch.save(V_model.state_dict(), os.path.join(log_path, f'V_finetune_enum_iter{i+1}.pth'))
    
    # Save final model
    torch.save(V_model.state_dict(), os.path.join(log_path, 'V_finetune_enum.pth'))
    writer.close()
    
    print(f"\nFine-tuning complete. Model saved to {log_path}")
    
    return log_path
