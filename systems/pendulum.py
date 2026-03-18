"""Pendulum system definition."""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are


class Pendulum:
    """
    Inverted pendulum system.
    
    State: [th, th_dot]
    - th: angle from vertical
    - th_dot: angular velocity
    
    Action: [T]
    - T: torque
    
    Constraint: Keep angle within safe bounds
    """
    
    # State space
    x_dim = 2
    x_low = np.array([-np.pi / 4, -4.0], dtype=np.float32)
    x_high = np.array([np.pi / 4, 4.0], dtype=np.float32)
    
    # Action space
    u_dim = 1
    u_low = np.array([-1.0], dtype=np.float32)
    u_high = np.array([1.0], dtype=np.float32)
    
    # Parameters
    th_con = np.pi / 6
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05 
    tmp1 = 3 * g / (2 * l) * dt
    tmp2 = 9 / (m * l ** 2) * dt
    
    @staticmethod
    def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute next state given current state and action.
        
        Args:
            x: State [2, N]
            u: Action [1, N]
            
        Returns:
            x_next: Next state [2, N]
        """
        th = x[0, :]
        thdot = x[1, :]
        T = u[0, :]
        
        newthdot = thdot + Pendulum.tmp1 * np.sin(th) + Pendulum.tmp2 * T
        newthdot = np.clip(newthdot, Pendulum.x_low[1], Pendulum.x_high[1])
        newth = th + newthdot * Pendulum.dt
        newth = np.clip(newth, Pendulum.x_low[0], Pendulum.x_high[0])
        
        x_next = np.stack([newth, newthdot], axis=0).astype(np.float32)
        return x_next
    
    @staticmethod
    def constraint(x: np.ndarray) -> np.ndarray:
        """
        Compute constraint value (negative = safe).
        
        Args:
            x: State [2, N]
            
        Returns:
            h: Constraint value [1, N] (h <= 0 means safe)
        """
        th = x[0, :]
        # max(-th - th_con, th - th_con)
        h = np.maximum(-th - Pendulum.th_con, th - Pendulum.th_con)
        return h.reshape(1, -1).astype(np.float32)
    
    @staticmethod
    def terminated(x: np.ndarray) -> np.ndarray:
        """
        Check if episode should terminate (state at boundary).
        
        Args:
            x: State [2, N]
            
        Returns:
            done: Boolean array [N]
        """
        th = x[0, :]
        return (th == Pendulum.x_low[0]) | (th == Pendulum.x_high[0])
    
    @staticmethod
    def create_policy() -> nn.Module:
        """
        Create LQR policy with state clipping.
        
        Returns:
            pi_model: Policy network
        """
        # LQR design
        A = np.array([[1 + Pendulum.tmp1 * Pendulum.dt, Pendulum.dt],
                      [Pendulum.tmp1, 1]], dtype=np.float32)
        B = np.array([[Pendulum.tmp2 * Pendulum.dt],
                      [Pendulum.tmp2]], dtype=np.float32)
        Q = np.diag([1.0, 0.1]).astype(np.float32)
        R = np.diag([0.1]).astype(np.float32)
        
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        K = K.astype(np.float32)
        
        # Policy: u = -K*x, clipped to [u_low, u_high]
        layers = []
        
        # Linear feedback
        linear = nn.Linear(2, 1)
        linear.weight.data = torch.from_numpy(-K)
        linear.bias.data = torch.zeros(1)
        layers.append(linear)
        
        # Clip to action bounds using ReLU trick
        # max(u, u_low) = relu(u - u_low) + u_low
        identity = nn.Linear(1, 1)
        identity.weight.data = torch.eye(1)
        identity.bias.data = torch.from_numpy(-Pendulum.u_low)
        layers.append(identity)
        layers.append(nn.ReLU())
        
        identity2 = nn.Linear(1, 1)
        identity2.weight.data = torch.eye(1)
        identity2.bias.data = torch.from_numpy(Pendulum.u_low)
        layers.append(identity2)
        
        # min(u, u_high) = -relu(-u + u_high) + u_high
        identity3 = nn.Linear(1, 1)
        identity3.weight.data = -torch.eye(1)
        identity3.bias.data = torch.from_numpy(Pendulum.u_high)
        layers.append(identity3)
        layers.append(nn.ReLU())
        
        identity4 = nn.Linear(1, 1)
        identity4.weight.data = -torch.eye(1)
        identity4.bias.data = torch.from_numpy(Pendulum.u_high)
        layers.append(identity4)
        
        pi_model = nn.Sequential(*layers)
        
        # Freeze parameters
        for param in pi_model.parameters():
            param.requires_grad = False
        
        return pi_model


# Alias for convenience
pi_model = Pendulum.create_policy()
