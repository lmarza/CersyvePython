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
        Create fixed policy pi(x) = [1, -yo].
        
        This policy accelerates forward and turns away from obstacle.
        
        Returns:
            pi_model: Policy network
        """
        # Linear layer: u = W*x + b
        # u[0] = 1 (constant acceleration)
        # u[1] = -yo (turn away from obstacle in y direction)
        weight = torch.tensor([[0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0]], dtype=torch.float32)
        bias = torch.tensor([1.0, 0.0], dtype=torch.float32)
        
        pi_model = nn.Linear(3, 2)
        pi_model.weight.data = weight
        pi_model.bias.data = bias
        
        # Freeze parameters
        for param in pi_model.parameters():
            param.requires_grad = False
        
        return pi_model


# Alias for convenience
pi_model = Unicycle.create_policy()
