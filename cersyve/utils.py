"""Utility functions for data collection, training, and regularization."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import Callable, Dict, Optional, Tuple
import os


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    # this line below was what fixed determinism for me before, when I was training RNNs, source https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def uniform(low: np.ndarray, high: np.ndarray, N: int) -> np.ndarray:
    """
    Generate N uniform random samples in the range [low, high].
    
    Args:
        low: Lower bounds (shape: [dim])
        high: Upper bounds (shape: [dim])
        N: Number of samples
        
    Returns:
        samples: Random samples (shape: [dim, N])
    """
    dim = len(low)
    return low.reshape(-1, 1) + np.random.rand(dim, N).astype(np.float32) * (high - low).reshape(-1, 1)


def collect_data(
    x_low: np.ndarray,
    x_high: np.ndarray,
    u_low: np.ndarray, 
    u_high: np.ndarray,
    dynamics: Callable,
    terminated: Callable,
    step_num: int = 1000000,
    max_ep_len: int = 100,
    noise_scale: float = 0.01,
    save_path: Optional[str] = None,
    seed: int = 0
) -> Dict[str, np.ndarray]:
    """
    Collect trajectory data by sampling random actions.
    
    Args:
        x_low, x_high: State space bounds
        u_low, u_high: Action space bounds
        dynamics: Function (x, u) -> x_next
        terminated: Function x -> bool (episode termination)
        step_num: Total number of steps to collect
        max_ep_len: Maximum episode length
        noise_scale: Noise scale for data preprocessing
        save_path: Path to save collected data (optional)
        seed: Random seed for reproducibility
        
    Returns:
        data: Dictionary with keys 'x', 'u', 'dx', and their statistics
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    x_dim = len(x_low)
    u_dim = len(u_low)
    
    data = {
        'x': np.zeros((x_dim, step_num), dtype=np.float32),
        'u': np.zeros((u_dim, step_num), dtype=np.float32),
        'dx': np.zeros((x_dim, step_num), dtype=np.float32),
    }
    
    x = uniform(x_low, x_high, 1)
    ep_step = 0
    
    for i in tqdm(range(step_num), desc="Collecting data"):
        u = uniform(u_low, u_high, 1)
        x_prime = dynamics(x, u)
        ep_step += 1
        
        data['x'][:, i] = x[:, 0]
        data['u'][:, i] = u[:, 0]
        data['dx'][:, i] = x_prime[:, 0] - x[:, 0]
        
        if terminated(x)[0] or ep_step == max_ep_len:
            x = uniform(x_low, x_high, 1)
            ep_step = 0
        else:
            x = x_prime
    
    # Preprocess data: normalize and add noise
    data_prep = {}
    for k, v in data.items():
        v_mean = v.mean(axis=1, keepdims=True)
        v_std = v.std(axis=1, keepdims=True)
        noise = noise_scale * np.random.randn(*v.shape).astype(np.float32)
        data_prep[k] = (v - v_mean) / v_std + noise
        data_prep[k + '_mean'] = v_mean[:, 0]
        data_prep[k + '_std'] = v_std[:, 0]
    
    if save_path:
        np.savez(save_path, **data_prep)
    
    return data_prep


def train_dynamics(
    data: Dict[str, np.ndarray],
    f_model: nn.Module,
    closed_loop: bool = False,
    lr: float = 1e-3,
    batch_size: int = 256,
    epoch_num: int = 100,
    weight_decay: float = 0.0,
    penalty: Optional[str] = None,  # "APA" / "SNR" / None
    noise_scale: float = 0.1,
    space_size: Optional[np.ndarray] = None,
    apa_coef: float = 0.01,
    snr_coef: Tuple[float, float] = (1e-3, 5e-4),
    log_dir: Optional[str] = None,
    device: str = 'cpu',
    seed: int = 42
) -> str:
    """
    Train dynamics model f(x,u) to predict dx.
    
    Args:
        data: Dictionary with 'x', 'u', 'dx' and their statistics
        f_model: Neural network model
        closed_loop: If True, use only x as input (policy is part of model)
        lr: Learning rate
        batch_size: Batch size
        epoch_num: Number of epochs
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
    set_seed(seed)
    f_model = f_model.to(device)
    
    if closed_loop:
        xu = torch.from_numpy(data['x']).to(device)
    else:
        xu = torch.from_numpy(np.vstack([data['x'], data['u']])).to(device)
    dx = torch.from_numpy(data['dx']).to(device)
    
    # Convert space_size to tensor if provided
    if space_size is not None:
        space_size_tensor = torch.from_numpy(space_size).to(device)
    else:
        space_size_tensor = None
    
    dataset = TensorDataset(xu.T, dx.T)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(f_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    
    if log_dir is None:
        log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"dynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_path)
    
    global_step = 0
    for epoch in tqdm(range(epoch_num), desc="Training dynamics"):
        losses = []
        mses = []
        
        for batch_xu, batch_dx in loader:
            # Data is already in [N, dim] shape from loader
            
            optimizer.zero_grad()
            
            if penalty is None:
                dx_pred = f_model(batch_xu)
                loss = torch.mean((dx_pred - batch_dx) ** 2)
            elif penalty == "SNR":
                noise = noise_scale * space_size_tensor.reshape(1, -1) / 2 * torch.randn_like(batch_xu)
                xu_noisy = torch.cat([batch_xu, batch_xu + noise], dim=0)
                dx_pred, snr_loss = forward_with_snr(f_model, xu_noisy, alpha=snr_coef[0], beta=snr_coef[1])
                mse_loss = torch.mean((dx_pred - batch_dx) ** 2)
                loss = mse_loss + snr_loss
            elif penalty == "APA":
                noise = noise_scale * space_size_tensor.reshape(1, -1) / 2 * torch.randn_like(batch_xu)
                xu_noisy = torch.cat([batch_xu, batch_xu + noise], dim=0)
                dx_pred, apa_loss = forward_with_apa(f_model, xu_noisy, alpha=apa_coef)
                mse_loss = torch.mean((dx_pred - batch_dx) ** 2)
                loss = mse_loss + apa_loss
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            with torch.no_grad():
                mses.append(torch.mean((f_model(batch_xu) - batch_dx) ** 2).item())
            
            global_step += 1
        
        writer.add_scalar('dynamics/loss', np.mean(losses), epoch)
        writer.add_scalar('dynamics/mean_squared_error', np.mean(mses), epoch)
    
    # Save model
    torch.save(f_model.state_dict(), os.path.join("../../models/double_integrator/", 'f.pth'))
    writer.close()
    
    return log_path


def train_constraint(
    x_low: np.ndarray,
    x_high: np.ndarray,
    h_model: nn.Module,
    constraint: Callable,
    lr: float = 3e-4,
    batch_size: int = 256,
    n_iter: int = 100000,
    log_dir: Optional[str] = None,
    device: str = 'cpu',
    seed: int = 42
) -> str:
    """
    Train constraint model h(x) to predict safety cost.
    
    Args:
        x_low, x_high: State space bounds
        h_model: Neural network model
        constraint: Function x -> h (constraint value)
        lr: Learning rate
        batch_size: Batch size
        n_iter: Number of iterations
        log_dir: Directory for tensorboard logs
        device: Device to train on
        seed: Random seed for reproducibility
        
    Returns:
        log_path: Path to saved model and logs
    """
    set_seed(seed)
    h_model = h_model.to(device)
    optimizer = torch.optim.Adam(h_model.parameters(), lr=lr)
    
    if log_dir is None:
        log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"constraint_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_path)
    
    for i in tqdm(range(n_iter), desc="Training constraint"):
        x = uniform(x_low, x_high, batch_size)
        c = constraint(x)
        
        x_t = torch.from_numpy(x.T).to(device)  # Transpose to [N, dim]
        c_t = torch.from_numpy(c.T).to(device)  # Transpose to [N, 1]
        
        optimizer.zero_grad()
        c_pred = h_model(x_t)
        loss = torch.mean((c_pred - c_t) ** 2)
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('constraint/loss', loss.item(), i)
        writer.add_scalar('constraint/constraint_satisfying_rate', float((c <= 0).mean()), i)
    
    # Save model
    torch.save(h_model.state_dict(), os.path.join("../../models/double_integrator/", 'h.pth'))
    writer.close()
    
    return log_path


def forward_with_snr(
    model: nn.Module,
    x: torch.Tensor,
    alpha: float = 1e-3,
    beta: float = 5e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass with Signal-to-Noise Ratio (SNR) regularization.
    
    Args:
        model: Neural network (Sequential with ReLU activations)
        x: Input tensor [2*batch_size, dim] (clean samples + noisy samples)
        alpha: Weight for variance loss
        beta: Weight for stability loss
        
    Returns:
        output: Model output on clean samples [batch_size, dim]
        snr_loss: SNR regularization loss
    """
    batch_size = x.shape[0] // 2
    snr_loss = 0.0
    
    for layer in model:
        if isinstance(layer, nn.Linear):
            x = layer(x)
            x_und = x[:batch_size, :]
            x_dis = x[batch_size:, :]
            
            # Variance loss
            var_loss = torch.mean(
                torch.sum((x_dis - x_und) ** 2, dim=1) / 
                torch.sum(x_und.detach() ** 2, dim=1).clamp(min=1e-8)
            )
            
            # Stability loss
            sta_loss = torch.mean(
                torch.sum(torch.abs(x_dis - x_und) / (x_und ** 2 + 1e-8), dim=1)
            )
            
            snr_loss += alpha * var_loss + beta * sta_loss
        elif isinstance(layer, nn.ReLU):
            x = layer(x)
    
    return x[:batch_size, :], snr_loss


def forward_with_apa(
    model: nn.Module,
    x: torch.Tensor,
    alpha: float = 0.01,
    eps: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass with Activation Pattern Alignment (APA) regularization.
    
    Args:
        model: Neural network (Sequential with ReLU activations)
        x: Input tensor [2*batch_size, dim] (clean samples + noisy samples)
        alpha: APA regularization coefficient
        eps: Small constant for numerical stability
        
    Returns:
        output: Model output on clean samples [batch_size, dim]
        apa_loss: APA regularization loss
    """
    batch_size = x.shape[0] // 2
    apa_loss = 0.0
    
    for layer in model:
        if isinstance(layer, nn.Linear):
            x = layer(x)
            x_und = x[:batch_size, :]
            x_dis = x[batch_size:, :]
            
            # APA loss: penalize misaligned activation patterns
            apa_loss += alpha * torch.mean(
                torch.sum(
                    torch.clamp(-x_und * x_dis, min=0) / 
                    torch.clamp(-x_und.detach() * x_dis.detach(), min=eps),
                    dim=1
                )
            )
        elif isinstance(layer, nn.ReLU):
            x = layer(x)
    
    return x[:batch_size, :], apa_loss


def get_activation_pattern(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Get activation patterns (binary masks) for ReLU neurons.
    
    Args:
        model: Neural network
        x: Input tensor [dim, N]
        
    Returns:
        patterns: Binary activation patterns [total_neurons, N]
    """
    patterns = []
    
    with torch.no_grad():
        for layer in model:
            if isinstance(layer, nn.Linear):
                z = layer(x.T).T
                # Check if next layer is ReLU
                patterns.append((z > 0).float())
                x = torch.relu(z)
    
    if patterns:
        return torch.cat(patterns, dim=0)
    return torch.empty(0, x.shape[1])


def count_activation_pattern(patterns: torch.Tensor) -> Tuple[list, list]:
    """
    Count unique activation patterns.
    
    Args:
        patterns: Binary activation patterns [neurons, N]
        
    Returns:
        log_points: Logarithmically spaced sample points
        uni_pat_num: Number of unique patterns at each log point
    """
    unique_patterns = set()
    log_points = [10 ** i for i in range(1, int(np.log10(patterns.shape[1])) + 1)]
    uni_pat_num = []
    
    for i in range(patterns.shape[1]):
        pattern_tuple = tuple(patterns[:, i].cpu().numpy())
        unique_patterns.add(pattern_tuple)
        
        if (i + 1) in log_points:
            uni_pat_num.append(len(unique_patterns))
    
    return log_points[:len(uni_pat_num)], uni_pat_num
