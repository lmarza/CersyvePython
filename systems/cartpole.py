"""Cart-pole system definition."""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are


class CartPole:
    """
    Cart-pole balancing system.
    
    State: [y, th, y_dot, th_dot]
    - y: cart position
    - th: pole angle from vertical
    - y_dot: cart velocity
    - th_dot: pole angular velocity
    
    Action: [F]
    - F: force on cart
    
    Constraint: Keep cart position and pole angle within safe bounds
    """
    
    # State space
    x_dim = 4
    x_high = np.array([1.0, 24 * np.pi / 180, 1.0, 1.0], dtype=np.float32)
    x_low = -x_high
    
    # Action space
    u_dim = 1
    u_high = np.array([1.0], dtype=np.float32)
    u_low = -u_high
    
    # Constraint bounds
    y_con = 0.8
    th_con = 12 * np.pi / 180
    
    # Parameters
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    lengthpole = 0.5
    polemass_length = masspole * lengthpole
    force_mag = 10.0
    dt = 0.05
    
    @staticmethod
    def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute next state given current state and action.
        
        Args:
            x: State [4, N]
            u: Action [1, N]
            
        Returns:
            x_next: Next state [4, N]
        """
        y, th, y_dot, th_dot = x[0, :], x[1, :], x[2, :], x[3, :]
        F = u[0, :]
        
        temp = (CartPole.force_mag * F + CartPole.polemass_length * th_dot ** 2 * np.sin(th)) / CartPole.total_mass
        thacc = (CartPole.gravity * np.sin(th) - np.cos(th) * temp) / (
            CartPole.lengthpole * (4.0 / 3.0 - CartPole.masspole * np.cos(th) ** 2 / CartPole.total_mass)
        )
        yacc = temp - CartPole.polemass_length * thacc * np.cos(th) / CartPole.total_mass
        
        x_next = np.stack([
            y + CartPole.dt * y_dot,
            th + CartPole.dt * th_dot,
            y_dot + CartPole.dt * yacc,
            th_dot + CartPole.dt * thacc,
        ], axis=0).astype(np.float32)
        
        # Clip to state bounds
        x_next = np.clip(x_next, CartPole.x_low.reshape(-1, 1), CartPole.x_high.reshape(-1, 1))
        
        return x_next
    
    @staticmethod
    def constraint(x: np.ndarray) -> np.ndarray:
        """
        Compute constraint value (negative = safe).
        
        Constraint: max(|y| - y_con, |th| - th_con)
        
        Args:
            x: State [4, N]
            
        Returns:
            h: Constraint value [1, N] (h <= 0 means safe)
        """
        y = x[0, :]
        th = x[1, :]
        
        # max(|y| - y_con, 0)
        h_y = np.maximum(np.abs(y) - CartPole.y_con, 0)
        
        # max(|th| - th_con, 0)
        h_th = np.maximum(np.abs(th) - CartPole.th_con, 0)
        
        # Combined constraint
        h = np.maximum(h_y - h_th, 0) + h_th
        
        return h.reshape(1, -1).astype(np.float32)
    
    @staticmethod
    def terminated(x: np.ndarray) -> np.ndarray:
        """
        Check if episode should terminate (state at boundary).
        
        Args:
            x: State [4, N]
            
        Returns:
            done: Boolean array [N]
        """
        y = x[0, :]
        th = x[1, :]
        return ((y == CartPole.x_low[0]) | (y == CartPole.x_high[0]) |
                (th == CartPole.x_low[1]) | (th == CartPole.x_high[1]))
    
    @staticmethod
    def create_policy() -> nn.Module:
        """
        Create LQR policy with state clipping.
        
        Returns:
            pi_model: Policy network
        """
        # Linearized dynamics
        temp = CartPole.lengthpole * (4.0 / 3.0 - CartPole.masspole / CartPole.total_mass)
        A = np.array([
            [1, 0, CartPole.dt, 0],
            [0, 1, 0, CartPole.dt],
            [0, 0, 1, -CartPole.dt * CartPole.polemass_length / CartPole.total_mass],
            [0, CartPole.dt * CartPole.gravity / temp, 0, 1],
        ], dtype=np.float32)
        B = np.array([
            [0],
            [0],
            [CartPole.dt * CartPole.force_mag / CartPole.total_mass],
            [-CartPole.dt * CartPole.force_mag / (CartPole.total_mass * temp)]
        ], dtype=np.float32)
        Q = np.diag([1.0, 10.0, 0.1, 1.0]).astype(np.float32)
        R = np.diag([0.1]).astype(np.float32)
        
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        
        # Policy: u = -K*x, clipped to [u_low, u_high]
        layers = []
        
        # Linear feedback
        linear = nn.Linear(4, 1)
        linear.weight.data = torch.from_numpy(-K)
        linear.bias.data = torch.zeros(1)
        layers.append(linear)
        
        # Clip to action bounds using ReLU trick
        # max(u, u_low) = relu(u - u_low) + u_low
        identity = nn.Linear(1, 1)
        identity.weight.data = torch.eye(1)
        identity.bias.data = torch.from_numpy(-CartPole.u_low)
        layers.append(identity)
        layers.append(nn.ReLU())
        
        identity2 = nn.Linear(1, 1)
        identity2.weight.data = torch.eye(1)
        identity2.bias.data = torch.from_numpy(CartPole.u_low)
        layers.append(identity2)
        
        # min(u, u_high) = -relu(-u + u_high) + u_high
        identity3 = nn.Linear(1, 1)
        identity3.weight.data = -torch.eye(1)
        identity3.bias.data = torch.from_numpy(CartPole.u_high)
        layers.append(identity3)
        layers.append(nn.ReLU())
        
        identity4 = nn.Linear(1, 1)
        identity4.weight.data = -torch.eye(1)
        identity4.bias.data = torch.from_numpy(CartPole.u_high)
        layers.append(identity4)
        
        pi_model = nn.Sequential(*layers)
        
        # Freeze parameters
        for param in pi_model.parameters():
            param.requires_grad = False
        
        return pi_model


# Alias for convenience
pi_model = CartPole.create_policy()
