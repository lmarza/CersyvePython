"""Pretrain value function using Bellman equation."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Tuple
import os

from .utils import uniform, forward_with_snr, forward_with_apa, set_seed


def pretrain_value(
    V_model: nn.Module,
    f_pi_model: nn.Module,
    h_model: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    gamma: float = 0.9,
    lr: float = 3e-4,
    batch_size: int = 256,
    iter_num: int = 100000,
    weight_decay: float = 0.0,
    penalty: Optional[str] = None,  # "APA" / "SNR" / None
    noise_scale: float = 0.1,
    space_size: Optional[np.ndarray] = None,
    apa_coef: float = 0.01,
    snr_coef: Tuple[float, float] = (1e-3, 5e-4),
    log_dir: Optional[str] = None,
    device: str = 'cpu',
    seed: int = 1
) -> str:
    """
    Pretrain value function using Bellman equation:
    V(x) = (1-γ)*h(x) + γ*max(h(x'), V(x'))
    
    Args:
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        h_model: Constraint model
        x_low, x_high: State space bounds
        gamma: Discount factor
        lr: Learning rate
        batch_size: Batch size
        iter_num: Number of iterations
        weight_decay: Weight decay coefficient
        penalty: Regularization method ("APA", "SNR", or None)
        noise_scale: Noise scale for regularization
        space_size: State space size for regularization
        apa_coef: APA regularization coefficient
        snr_coef: SNR regularization coefficients (alpha, beta)
        log_dir: Directory for tensorboard logs
        device: Device to train on
        seed: Random seed for reproducibility
    Returns:
        log_path: Path to saved model and logs
    """
    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    h_model = h_model.to(device)
    
    optimizer = torch.optim.AdamW(V_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    
    if log_dir is None:
        log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_path)
    
    set_seed(seed)
    
    for i in tqdm(range(iter_num), desc="Pretraining value function"):
        # Sample random states
        x = uniform(x_low, x_high, batch_size)
        x_t = torch.from_numpy(x).to(device)
        
        with torch.no_grad():
            # Compute next state and constraint
            x_prime_t = f_pi_model(x_t)
            c_t = h_model(x_t.T).T
            v_prime_t = V_model(x_prime_t.T).T
            
            # Bellman target: (1-γ)*h(x) + γ*max(h(x'), V(x'))
            v_targ = (1 - gamma) * c_t + gamma * torch.maximum(c_t, v_prime_t)
        
        optimizer.zero_grad()
        
        if penalty is None:
            v_pred = V_model(x_t.T).T
            loss = torch.mean((v_pred - v_targ) ** 2)
        elif penalty == "SNR":
            noise = noise_scale * space_size.reshape(-1, 1) / 2 * np.random.randn(*x.shape).astype(np.float32)
            x_noisy = np.hstack([x, x + noise])
            x_noisy_t = torch.from_numpy(x_noisy.T).to(device)  # Transpose to [2*N, dim]
            v_pred, snr_loss = forward_with_snr(V_model, x_noisy_t, alpha=snr_coef[0], beta=snr_coef[1])
            mse_loss = torch.mean((v_pred.T - v_targ) ** 2)
            loss = mse_loss + snr_loss
        elif penalty == "APA":
            noise = noise_scale * space_size.reshape(-1, 1) / 2 * np.random.randn(*x.shape).astype(np.float32)
            x_noisy = np.hstack([x, x + noise])
            x_noisy_t = torch.from_numpy(x_noisy.T).to(device)  # Transpose to [2*N, dim]
            v_pred, apa_loss = forward_with_apa(V_model, x_noisy_t, alpha=apa_coef)
            mse_loss = torch.mean((v_pred.T - v_targ) ** 2)
            loss = mse_loss + apa_loss
        
        loss.backward()
        optimizer.step()
        
        # Logging
        with torch.no_grad():
            constraint_satisfying = (c_t <= 0).float().mean().item()
            predicted_feasible = (V_model(x_t.T).T <= 0).float().mean().item()
        
        writer.add_scalar('pretrain/loss', loss.item(), i)
        writer.add_scalar('pretrain/constraint_satisfying_rate', constraint_satisfying, i)
        writer.add_scalar('pretrain/predicted_feasible_rate', predicted_feasible, i)
    
    # Save model
    torch.save(V_model.state_dict(), os.path.join("../../models/double_integrator/", 'V_pretrain.pth'))
    writer.close()
    
    return log_path
