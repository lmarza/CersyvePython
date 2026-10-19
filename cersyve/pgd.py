"""Adversarial search algorithms for finding counterexamples."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def boundary_guided_search(
    x: np.ndarray,
    x_low: np.ndarray,
    x_high: np.ndarray,
    h_model: nn.Module,
    V_model: nn.Module,
    f_pi_model: nn.Module,
    pgd_step: int = 10,
    pgd_eps: float = 0.1,
    pgd_beta: float = 0.0,
    backtrack_step: int = 20,
    length_discount: float = 0.8,
    bound_guide: bool = True,
    direct_discount: float = 0.5,
    tol: float = 1e-4,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Boundary-Guided Branching (BGB) search for counterexamples.
    
    Performs projected gradient descent along the boundary of the feasible region
    to find states that violate safety constraints.
    
    Args:
        x: Initial states [dim, N]
        x_low, x_high: State space bounds
        h_model: Constraint model
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        pgd_step: Number of PGD steps
        pgd_eps: Step size for PGD
        pgd_beta: Momentum coefficient
        backtrack_step: Number of backtracking steps
        length_discount: Discount factor for backtracking step length
        bound_guide: Whether to use boundary guidance
        direct_discount: Discount factor for boundary direction
        tol: Tolerance for feasibility
        device: Device to run on
        
    Returns:
        x: Updated states [dim, N]
    """
    h_model = h_model.to(device)
    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    
    pgd_mask = np.ones(x.shape[1], dtype=bool)
    x_pgd = x.copy().astype(np.float32)
    momentum = np.zeros_like(x, dtype=np.float32)
    
    for _ in range(pgd_step):
        x_t = torch.from_numpy(x_pgd.astype(np.float32)).to(device).requires_grad_(True)
        
        with torch.enable_grad():
            h = h_model(x_t.T).T
            v = V_model(x_t.T).T
            x_next = f_pi_model(x_t)
            v_prime = V_model(x_next.T).T
        
        h_np = h.detach().cpu().numpy()[0, :]
        v_np = v.detach().cpu().numpy()[0, :]
        v_prime_np = v_prime.detach().cpu().numpy()[0, :]
        
        # Check for constraint and invariance counterexamples
        con = (v_np <= tol) & (h_np > -tol)
        inv = (v_np <= tol) & (v_prime_np > -tol)
        pgd_mask_pgd = pgd_mask[pgd_mask]
        pgd_mask_pgd[con | inv] = False
        pgd_mask[pgd_mask] = pgd_mask_pgd
        
        if not np.any(pgd_mask):
            break
        
        x_pgd = x[:, pgd_mask].astype(np.float32)
        x_t = torch.from_numpy(x_pgd).to(device).requires_grad_(True)
        
        # Compute gradients for constraint and invariance
        n_con = x_pgd.shape[1] // 2
        
        # Constraint gradient
        h_con = h_model(x_t[:, :n_con].T).T
        h_con.sum().backward()
        con_g = x_t.grad[:, :n_con].clone()
        x_t.grad.zero_()
        
        # Invariance gradient
        x_next_inv = f_pi_model(x_t[:, n_con:])
        v_prime_inv = V_model(x_next_inv.T).T
        v_prime_inv.sum().backward()
        inv_g = x_t.grad[:, n_con:].clone()
        
        g = torch.cat([con_g, inv_g], dim=1).cpu().numpy()
        g += pgd_beta * momentum[:, pgd_mask]
        g /= np.linalg.norm(g, axis=0, keepdims=True) + 1e-8
        
        # Value function gradient (for boundary direction)
        x_t = torch.from_numpy(x_pgd.astype(np.float32)).to(device).requires_grad_(True)
        v_pgd = V_model(x_t.T).T
        v_pgd.sum().backward()
        v_g = x_t.grad.cpu().numpy()
        v_g /= np.linalg.norm(v_g, axis=0, keepdims=True) + 1e-8
        
        # Compute boundary-orthogonal direction
        a = np.sum(g * v_g, axis=0, keepdims=True)
        z = a * g - v_g
        z /= np.linalg.norm(z, axis=0, keepdims=True) + 1e-8
        
        # Backtracking line search
        direction = np.zeros_like(x_pgd)
        coef = np.zeros(x_pgd.shape[1])
        active = np.ones(x_pgd.shape[1], dtype=bool)
        
        for i in range(backtrack_step + 1):
            if bound_guide:
                d_coef = direct_discount ** i
                d = d_coef * g[:, active] + (1 - d_coef) * z[:, active]
                d /= np.linalg.norm(d, axis=0, keepdims=True) + 1e-8
            else:
                d = g[:, active]
            
            l_coef = length_discount ** i
            x_tmp = x_pgd[:, active] + l_coef * pgd_eps * d
            x_tmp = np.clip(x_tmp, x_low.reshape(-1, 1), x_high.reshape(-1, 1))
            
            # Check feasibility
            x_tmp_t = torch.from_numpy(x_tmp).to(device)
            with torch.no_grad():
                v_tmp = V_model(x_tmp_t.T).T.cpu().numpy()[0, :]
            fea = v_tmp <= tol
            
            direction[:, active] = d
            coef[active] = np.where(fea, l_coef, coef[active])
            active[active] = ~fea
            
            if not np.any(active):
                break
        
        # Update states
        dx = coef.reshape(1, -1) * direction
        x_pgd = x_pgd + pgd_eps * dx
        x_pgd = np.clip(x_pgd, x_low.reshape(-1, 1), x_high.reshape(-1, 1))
        x[:, pgd_mask] = x_pgd
        
        momentum_pgd = momentum[:, pgd_mask]
        momentum_pgd[:, ~active] = dx[:, ~active]
        momentum[:, pgd_mask] = momentum_pgd
    
    return x


def projected_boundary_search(
    x: np.ndarray,
    x_low: np.ndarray,
    x_high: np.ndarray,
    h_model: nn.Module,
    V_model: nn.Module,
    f_pi_model: nn.Module,
    lr: float = 0.01,
    grad_step: int = 10,
    proj_step: int = 10,
    proj_tol: float = 0.1,
    tol: float = 1e-4,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Projected Boundary Search (PBS) for counterexamples.
    
    Projects gradient ascent onto the boundary manifold defined by V(x) = 0.
    
    Args:
        x: Initial states [dim, N]
        x_low, x_high: State space bounds
        h_model: Constraint model
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        lr: Learning rate
        grad_step: Number of gradient steps
        proj_step: Number of projection steps
        proj_tol: Tolerance for projection to boundary
        tol: Tolerance for feasibility
        device: Device to run on
        
    Returns:
        x: Updated states [dim, N]
    """
    h_model = h_model.to(device)
    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    
    pgd_mask = np.ones(x.shape[1], dtype=bool)
    x_pgd = x.copy()
    
    for _ in range(grad_step):
        x_t = torch.from_numpy(x_pgd).to(device).requires_grad_(True)
        
        with torch.enable_grad():
            h = h_model(x_t.T).T
            v = V_model(x_t.T).T
            x_next = f_pi_model(x_t)
            v_prime = V_model(x_next.T).T
        
        h_np = h.detach().cpu().numpy()[0, :]
        v_np = v.detach().cpu().numpy()[0, :]
        v_prime_np = v_prime.detach().cpu().numpy()[0, :]
        
        # Check for counterexamples
        con = (v_np <= tol) & (h_np > -tol)
        inv = (v_np <= tol) & (v_prime_np > -tol)
        pgd_mask_pgd = pgd_mask[pgd_mask]
        pgd_mask_pgd[con | inv] = False
        pgd_mask[pgd_mask] = pgd_mask_pgd
        
        if not np.any(pgd_mask):
            break
        
        x_pgd = x[:, pgd_mask]
        x_t = torch.from_numpy(x_pgd).to(device).requires_grad_(True)
        
        # Compute constraint and invariance gradients
        n_con = x_pgd.shape[1] // 2
        
        h_con = h_model(x_t[:, :n_con].T).T
        h_con.sum().backward()
        con_g = x_t.grad[:, :n_con].clone()
        x_t.grad.zero_()
        
        x_next_inv = f_pi_model(x_t[:, n_con:])
        v_prime_inv = V_model(x_next_inv.T).T
        v_prime_inv.sum().backward()
        inv_g = x_t.grad[:, n_con:].clone()
        
        g = torch.cat([con_g, inv_g], dim=1)
        
        # Value gradient for projection
        x_t = torch.from_numpy(x_pgd).to(device).requires_grad_(True)
        v_pgd = V_model(x_t.T).T
        v_pgd.sum().backward()
        v_g = x_t.grad
        v_g_norm = v_g / (torch.norm(v_g, dim=0, keepdim=True) + 1e-8)
        
        # Project gradient onto boundary tangent space
        proj_g = g - torch.sum(g * v_g_norm, dim=0, keepdim=True) * v_g_norm
        
        # Update samples
        x_pgd = x_pgd + lr * proj_g.cpu().numpy()
        x_pgd = np.clip(x_pgd, x_low.reshape(-1, 1), x_high.reshape(-1, 1))
        
        # Project samples back to boundary
        proj_mask = np.ones(x_pgd.shape[1], dtype=bool)
        for _ in range(proj_step):
            x_tmp_t = torch.from_numpy(x_pgd[:, proj_mask]).to(device).requires_grad_(True)
            v_tmp = V_model(x_tmp_t.T).T
            
            projected = torch.abs(v_tmp[0, :]).detach().cpu().numpy() <= proj_tol
            proj_mask_tmp = proj_mask[proj_mask]
            proj_mask_tmp[projected] = False
            proj_mask[proj_mask] = proj_mask_tmp
            
            if not np.any(proj_mask):
                break
            
            # Gradient descent to minimize |V(x)|
            v_abs = torch.abs(v_tmp)
            v_abs.sum().backward()
            d_tmp = x_tmp_t.grad.cpu().numpy()
            
            x_pgd[:, proj_mask] = x_pgd[:, proj_mask] - lr * d_tmp[:, proj_mask]
            x_pgd[:, proj_mask] = np.clip(
                x_pgd[:, proj_mask],
                x_low.reshape(-1, 1),
                x_high.reshape(-1, 1)
            )
        
        x[:, pgd_mask] = x_pgd
    
    return x


def filter_counterexample(
    x: np.ndarray,
    h_model: nn.Module,
    V_model: nn.Module,
    f_pi_model: nn.Module,
    tol: float = 1e-4,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter counterexamples into constraint and invariance violations.
    
    Args:
        x: States [dim, N]
        h_model: Constraint model
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        tol: Tolerance for feasibility
        device: Device to run on
        
    Returns:
        con: Boolean array for constraint violations [N]
        inv: Boolean array for invariance violations [N]
    """
    # The models already live on `device` (the training loops move them once);
    # the masks are formed there and copied back in a single transfer.
    x_t = torch.as_tensor(x, device=device)

    with torch.no_grad():
        h = h_model(x_t.T).T[0, :]
        v = V_model(x_t.T).T[0, :]
        x_prime = f_pi_model(x_t)
        v_prime = V_model(x_prime.T).T[0, :]

        # Constraint counterexample: V(x) <= 0 AND h(x) > 0
        con = (v <= tol) & (h > -tol)

        # Invariance counterexample: V(x) <= 0 AND V(x') > 0 (and not constraint violation)
        inv = (v <= tol) & (v_prime > -tol) & (~con)

        both = torch.stack([con, inv]).cpu().numpy()

    return both[0], both[1]
