"""Neural network model architectures."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional


def create_mlp(input_dim: int, output_dim: int, hidden_sizes: List[int]) -> nn.Sequential:
    """
    Create a Multi-Layer Perceptron (MLP) with ReLU activations.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_sizes: List of hidden layer sizes
        
    Returns:
        model: Sequential MLP model
    """
    layers = []
    
    # First layer
    layers.append(nn.Linear(input_dim, hidden_sizes[0]))
    layers.append(nn.ReLU())
    
    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(nn.ReLU())
    
    # Output layer (no activation)
    layers.append(nn.Linear(hidden_sizes[-1], output_dim))
    
    return nn.Sequential(*layers)


class ClosedLoopDynamicsModel(nn.Module):
    """
    Closed-loop dynamics model f_pi(x) = x + f(x, pi(x)) with normalization and state clipping.
    """
    def __init__(
        self,
        f_model: nn.Module,
        pi_model: nn.Module,
        data: Dict[str, np.ndarray],
        x_low: np.ndarray,
        x_high: np.ndarray,
        u_dim: int
    ):
        """
        Args:
            f_model: Dynamics model f(x,u) -> dx
            pi_model: Policy model pi(x) -> u
            data: Data dictionary with normalization statistics
            x_low, x_high: State space bounds
            u_dim: Action dimension
        """
        super().__init__()
        x_dim = len(x_low)
        
        self.f_model = f_model
        self.pi_model = pi_model
        
        # Normalization parameters
        self.register_buffer('x_mean', torch.from_numpy(data['x_mean']))
        self.register_buffer('x_std', torch.from_numpy(data['x_std']))
        self.register_buffer('u_mean', torch.from_numpy(data['u_mean']))
        self.register_buffer('u_std', torch.from_numpy(data['u_std']))
        self.register_buffer('dx_mean', torch.from_numpy(data['dx_mean']))
        self.register_buffer('dx_std', torch.from_numpy(data['dx_std']))
        
        # State bounds
        self.register_buffer('x_low', torch.from_numpy(x_low))
        self.register_buffer('x_high', torch.from_numpy(x_high))
        
        self.x_dim = x_dim
        self.u_dim = u_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State tensor [x_dim, N]
            
        Returns:
            x_next: Next state [x_dim, N]
        """
        # Normalize state
        x_norm = (x - self.x_mean.view(-1, 1)) / self.x_std.view(-1, 1)
        
        # Get action from policy and normalize
        u = self.pi_model(x.T).T
        u_norm = (u - self.u_mean.view(-1, 1)) / self.u_std.view(-1, 1)
        
        # Concatenate normalized state and action
        xu_norm = torch.cat([x_norm, u_norm], dim=0)
        
        # Predict dx and denormalize
        dx = self.f_model(xu_norm.T).T * self.dx_std.view(-1, 1) + self.dx_mean.view(-1, 1)
        
        # Update state: x_next = x + dx
        x_next = x + dx
        
        # Clip to state bounds using ReLU trick
        # max(x, x_low) = relu(x - x_low) + x_low
        x_next = torch.relu(x_next - self.x_low.view(-1, 1)) + self.x_low.view(-1, 1)
        # min(x, x_high) = -relu(-x + x_high) + x_high
        x_next = -torch.relu(-x_next + self.x_high.view(-1, 1)) + self.x_high.view(-1, 1)
        
        return x_next


def create_closed_loop_dynamics_model(
    f_model: nn.Module,
    pi_model: nn.Module,
    data: Dict[str, np.ndarray],
    x_low: np.ndarray,
    x_high: np.ndarray,
    u_dim: int
) -> ClosedLoopDynamicsModel:
    """
    Create closed-loop dynamics model.
    
    Args:
        f_model: Dynamics model f(x,u) -> dx
        pi_model: Policy model pi(x) -> u
        data: Data dictionary with normalization statistics
        x_low, x_high: State space bounds
        u_dim: Action dimension
        
    Returns:
        f_pi_model: Closed-loop dynamics model
    """
    return ClosedLoopDynamicsModel(f_model, pi_model, data, x_low, x_high, u_dim)


def create_linear_closed_loop_model(
    A: np.ndarray,
    B: np.ndarray,
    pi_model: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    drift: Optional[np.ndarray] = None,
) -> ClosedLoopDynamicsModel:
    """
    Exact closed-loop model for a linear system x' = clip(Ax + B pi(x)),
    encoded as a ClosedLoopDynamicsModel with identity normalization so it is
    directly usable by the MILP encoder and the training loops.
    """
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    x_dim, u_dim = A.shape[0], B.shape[1]

    # dx = (A - I)x + Bu, so x_next = x + dx
    f_linear = nn.Linear(x_dim + u_dim, x_dim)
    f_linear.weight.data = torch.from_numpy(
        np.concatenate([A - np.eye(x_dim, dtype=np.float32), B], axis=1)
    )
    if drift is None:
        f_linear.bias.data = torch.zeros(x_dim)
    else:
        f_linear.bias.data = torch.from_numpy(np.asarray(drift, dtype=np.float32))
    f_model = nn.Sequential(f_linear)

    identity_stats = {
        'x_mean': np.zeros(x_dim, dtype=np.float32),
        'x_std': np.ones(x_dim, dtype=np.float32),
        'u_mean': np.zeros(u_dim, dtype=np.float32),
        'u_std': np.ones(u_dim, dtype=np.float32),
        'dx_mean': np.zeros(x_dim, dtype=np.float32),
        'dx_std': np.ones(x_dim, dtype=np.float32),
    }
    model = ClosedLoopDynamicsModel(
        f_model, pi_model, identity_stats,
        x_low.astype(np.float32), x_high.astype(np.float32), u_dim,
    )
    for param in model.parameters():
        param.requires_grad = False
    return model


def create_box_constraint_model(
    x_dim: int,
    limits: Dict[int, float],
) -> nn.Sequential:
    """
    Exact piecewise-linear model of h(x) = max_i(|x_i| - c_i) over the given
    {dim: c} limits, as a Linear/ReLU Sequential (MILP-encodable).

    Uses |x| = relu(x) + relu(-x) and, for two limits,
    max(p, q) = p + relu(q - p) with p >= 0 arranged by construction.
    """
    dims = sorted(limits)
    if len(dims) not in (1, 2):
        raise ValueError("supported: one or two constrained dimensions")

    l1 = nn.Linear(x_dim, 2 * len(dims))
    l1.weight.data = torch.zeros(2 * len(dims), x_dim)
    for i, d in enumerate(dims):
        l1.weight.data[2 * i, d] = 1.0
        l1.weight.data[2 * i + 1, d] = -1.0
    l1.bias.data = torch.zeros(2 * len(dims))

    if len(dims) == 1:
        c = limits[dims[0]]
        l2 = nn.Linear(2, 1)
        l2.weight.data = torch.tensor([[1.0, 1.0]])
        l2.bias.data = torch.tensor([-c])
        model = nn.Sequential(l1, nn.ReLU(), l2)
    else:
        c0, c1 = limits[dims[0]], limits[dims[1]]
        l2 = nn.Linear(4, 2)
        l2.weight.data = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, 1.0, 1.0],
        ])
        l2.bias.data = torch.tensor([0.0, c0 - c1])
        l3 = nn.Linear(2, 1)
        l3.weight.data = torch.tensor([[1.0, 1.0]])
        l3.bias.data = torch.tensor([-c0])
        model = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3)

    for param in model.parameters():
        param.requires_grad = False
    return model


class ValueConstraintModel(nn.Module):
    """
    Model that outputs [V(x), h(x)] for verification.
    """
    def __init__(self, V_model: nn.Module, h_model: nn.Module):
        super().__init__()
        self.V_model = V_model
        self.h_model = h_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State tensor [x_dim, N]
            
        Returns:
            output: [V(x), h(x)] tensor [2, N]
        """
        v = self.V_model(x.T).T  # [1, N]
        h = self.h_model(x.T).T  # [1, N]
        return torch.cat([v, h], dim=0)  # [2, N]


def create_value_constraint_model(V_model: nn.Module, h_model: nn.Module) -> ValueConstraintModel:
    """
    Create value-constraint model for verification.
    
    Args:
        V_model: Value function model
        h_model: Constraint model
        
    Returns:
        model: Combined model outputting [V(x), h(x)]
    """
    return ValueConstraintModel(V_model, h_model)


class ValueNextValueModel(nn.Module):
    """
    Model that outputs [V(x), V(f_pi(x))] for verification.
    """
    def __init__(self, V_model: nn.Module, f_pi_model: nn.Module):
        super().__init__()
        self.V_model = V_model
        self.f_pi_model = f_pi_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State tensor [x_dim, N]
            
        Returns:
            output: [V(x), V(f_pi(x))] tensor [2, N]
        """
        v = self.V_model(x.T).T  # [1, N]
        x_next = self.f_pi_model(x)
        v_next = self.V_model(x_next.T).T  # [1, N]
        return torch.cat([v, v_next], dim=0)  # [2, N]


def create_value_next_value_model(V_model: nn.Module, f_pi_model: nn.Module) -> ValueNextValueModel:
    """
    Create value-next-value model for verification.
    
    Args:
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        
    Returns:
        model: Combined model outputting [V(x), V(f_pi(x))]
    """
    return ValueNextValueModel(V_model, f_pi_model)
