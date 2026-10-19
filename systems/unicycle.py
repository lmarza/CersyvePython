"""Unicycle system definition."""

import numpy as np
import torch
import torch.nn as nn


class Unicycle:
    """
    Unicycle obstacle avoidance system.
    
    State: [v, xo, yo]
    - v: velocity
    - xo, yo: obstacle position in robot frame
    
    Action: [a, w]
    - a: acceleration
    - w: angular velocity
    
    Constraint: Stay away from obstacle (distance > ro)
    """
    
    # State space
    x_dim = 3
    x_low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    x_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    # Action space
    u_dim = 2
    u_low = np.array([-1.0, -1.0], dtype=np.float32)
    u_high = np.array([1.0, 1.0], dtype=np.float32)
    
    # Parameters
    dt = 0.1
    ro = 0.4  # Obstacle radius
    
    @staticmethod
    def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute next state given current state and action.
        
        Args:
            x: State [3, N]
            u: Action [2, N]
            
        Returns:
            x_next: Next state [3, N]
        """
        v = x[0, :]
        xo = x[1, :]
        yo = x[2, :]
        
        a = u[0, :]
        w = u[1, :]
        
        # Update obstacle position in robot frame (robot moves)
        xo_tmp = xo - v * Unicycle.dt
        dth = w * Unicycle.dt
        
        x_next = np.stack([
            v + a * Unicycle.dt,
            xo_tmp * np.cos(dth) + yo * np.sin(dth),
            yo * np.cos(dth) - xo_tmp * np.sin(dth),
        ], axis=0).astype(np.float32)
        
        # Clip to state bounds
        x_next = np.clip(x_next, Unicycle.x_low.reshape(-1, 1), Unicycle.x_high.reshape(-1, 1))
        
        return x_next
    
    @staticmethod
    def constraint(x: np.ndarray) -> np.ndarray:
        """
        Compute constraint value (negative = safe).
        
        Args:
            x: State [3, N]
            
        Returns:
            h: Constraint value [1, N] (h <= 0 means safe)
        """
        xo = x[1, :]
        yo = x[2, :]
        dist = np.sqrt(xo ** 2 + yo ** 2)
        return (Unicycle.ro - dist).reshape(1, -1).astype(np.float32)
    
    @staticmethod
    def terminated(x: np.ndarray) -> np.ndarray:
        """
        Check if episode should terminate (state at boundary).
        
        Args:
            x: State [3, N]
            
        Returns:
            done: Boolean array [N]
        """
        xo = x[1, :]
        yo = x[2, :]
        return ((xo == Unicycle.x_low[1]) | (xo == Unicycle.x_high[1]) |
                (yo == Unicycle.x_low[2]) | (yo == Unicycle.x_high[2]))
    
    @staticmethod
    def create_policy() -> nn.Module:
        """
        Create fixed policy pi(x) = clip([1, -yo], u_low, u_high).

        This policy accelerates forward and turns away from obstacle.
        The output is clipped to action bounds using a ReLU trick so that
        the policy is representable as an nn.Sequential of Linear/ReLU layers,
        which is required by the MILP encoder.

        Returns:
            pi_model: Policy network (nn.Sequential)
        """
        # Linear layer: u = W*x + b
        # u[0] = 1 (constant acceleration)
        # u[1] = -yo (turn away from obstacle in y direction)
        weight = torch.tensor([[0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0]], dtype=torch.float32)
        bias = torch.tensor([1.0, 0.0], dtype=torch.float32)

        u_low = torch.from_numpy(Unicycle.u_low)   # [-1, -1]
        u_high = torch.from_numpy(Unicycle.u_high)  # [ 1,  1]

        layers = []

        # Base linear feedback
        linear = nn.Linear(3, 2)
        linear.weight.data = weight
        linear.bias.data = bias
        layers.append(linear)

        # clip lower: max(u, u_low) = relu(u - u_low) + u_low
        low_shift = nn.Linear(2, 2)
        low_shift.weight.data = torch.eye(2)
        low_shift.bias.data = -u_low
        layers.append(low_shift)
        layers.append(nn.ReLU())

        low_restore = nn.Linear(2, 2)
        low_restore.weight.data = torch.eye(2)
        low_restore.bias.data = u_low
        layers.append(low_restore)

        # clip upper: min(u, u_high) = -relu(-u + u_high) + u_high
        high_shift = nn.Linear(2, 2)
        high_shift.weight.data = -torch.eye(2)
        high_shift.bias.data = u_high
        layers.append(high_shift)
        layers.append(nn.ReLU())

        high_restore = nn.Linear(2, 2)
        high_restore.weight.data = -torch.eye(2)
        high_restore.bias.data = u_high
        layers.append(high_restore)

        pi_model = nn.Sequential(*layers)

        # Freeze parameters
        for param in pi_model.parameters():
            param.requires_grad = False

        return pi_model


# Alias for convenience
pi_model = Unicycle.create_policy()
