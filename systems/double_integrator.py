"""Double integrator system definition."""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are


class DoubleIntegrator:
    """
    Double integrator system.
    
    State: [x1, x2]
    - x1: position
    - x2: velocity
    
    Action: [u]
    - u: acceleration
    
    Constraint: Keep position within safe bounds (-x1_con, x1_con)
    """
    
    # State space
    x_dim = 2
    x_high = np.array([1.0, 1.0], dtype=np.float32)
    x_low = np.array([-1.0, -1.0], dtype=np.float32)
    
    # Action space
    u_dim = 1
    u_high = np.array([1.0], dtype=np.float32)
    u_low = np.array([-1.0], dtype=np.float32)
    
    # Constraint bound
    x1_con = 0.9
    
    # Time step
    dt = 0.1
    
    # Linear system matrices
    A = np.array([
        [1, dt],
        [0, 1]
    ], dtype=np.float32)
    
    B = np.array([
        [0],
        [dt]
    ], dtype=np.float32)
    
    # LQR cost matrices
    Q = np.diag([1.0, 0.1]).astype(np.float32)
    R = np.diag([0.1]).astype(np.float32)
    
    # Solve discrete-time Algebraic Riccati Equation
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    K = K.astype(np.float32)  # Ensure float32 dtype
    
    @staticmethod
    def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute next state given current state and action.
        
        State update: x' = Ax + Bu
        
        Args:
            x: State [2, N]
            u: Action [1, N]
            
        Returns:
            x_next: Next state [2, N]
        """
        x_next = DoubleIntegrator.A @ x + DoubleIntegrator.B @ u
        
        # Clip to state bounds
        x_next = np.clip(x_next, DoubleIntegrator.x_low.reshape(-1, 1), 
                        DoubleIntegrator.x_high.reshape(-1, 1))
        
        return x_next.astype(np.float32)
    
    @staticmethod
    def constraint(x: np.ndarray) -> np.ndarray:
        """
        Compute constraint value (negative = safe).
        
        Constraint: |x1| - x1_con
        The constraint is violated when x1 goes outside [-x1_con, x1_con]
        
        Args:
            x: State [2, N]
            
        Returns:
            h: Constraint value [1, N] (h <= 0 means safe)
        """
        x1 = x[0, :]
        
        # max(-x1_con - x1, x1 - x1_con)
        h = np.maximum(-DoubleIntegrator.x1_con - x1, x1 - DoubleIntegrator.x1_con)
        
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
        return ((x[0, :] == DoubleIntegrator.x_low[0]) | (x[0, :] == DoubleIntegrator.x_high[0]) |
                (x[1, :] == DoubleIntegrator.x_low[1]) | (x[1, :] == DoubleIntegrator.x_high[1]))
    
    @staticmethod
    def create_policy() -> nn.Module:
        """
        Create LQR policy with action clipping.
        
        Policy: u = -K*x, clipped to [u_low, u_high]
        
        Returns:
            pi_model: Policy network
        """
        layers = []
        
        # Linear feedback: u = -K*x
        linear = nn.Linear(2, 1)
        linear.weight.data = torch.from_numpy(-DoubleIntegrator.K)
        linear.bias.data = torch.zeros(1)
        layers.append(linear)
        
        # Clip to action bounds using ReLU trick
        # max(u, u_low) = relu(u - u_low) + u_low
        identity = nn.Linear(1, 1)
        identity.weight.data = torch.eye(1)
        identity.bias.data = torch.from_numpy(-DoubleIntegrator.u_low)
        layers.append(identity)
        layers.append(nn.ReLU())
        
        identity2 = nn.Linear(1, 1)
        identity2.weight.data = torch.eye(1)
        identity2.bias.data = torch.from_numpy(DoubleIntegrator.u_low)
        layers.append(identity2)
        
        # min(u, u_high) = -relu(-u + u_high) + u_high
        identity3 = nn.Linear(1, 1)
        identity3.weight.data = -torch.eye(1)
        identity3.bias.data = torch.from_numpy(DoubleIntegrator.u_high)
        layers.append(identity3)
        layers.append(nn.ReLU())
        
        identity4 = nn.Linear(1, 1)
        identity4.weight.data = -torch.eye(1)
        identity4.bias.data = torch.from_numpy(DoubleIntegrator.u_high)
        layers.append(identity4)
        
        pi_model = nn.Sequential(*layers)
        
        # Freeze parameters
        for param in pi_model.parameters():
            param.requires_grad = False
        
        return pi_model
    
    @staticmethod
    def create_constraint_model() -> nn.Module:
        """
        Create constraint neural network model.
        
        The constraint model combines two operations:
        1. A ReLU layer: relu(-2*x1)
        2. An affine layer: x1 - x1_con
        
        Returns:
            h_model: Constraint network
        """
        # First branch: Dense([-2, 0], [0], relu)
        # Output: relu(-2*x1 + 0)
        branch1_linear = nn.Linear(2, 1)
        branch1_linear.weight.data = torch.tensor([[-2.0, 0.0]])
        branch1_linear.bias.data = torch.zeros(1)
        branch1 = nn.Sequential(branch1_linear, nn.ReLU())
        
        # Second branch: Dense([1, 0], [-x1_con])
        # Output: x1 - x1_con
        branch2_linear = nn.Linear(2, 1)
        branch2_linear.weight.data = torch.tensor([[1.0, 0.0]])
        branch2_linear.bias.data = torch.tensor([-DoubleIntegrator.x1_con])
        branch2 = branch2_linear
        
        # Combine branches with addition
        class ParallelAdd(nn.Module):
            def __init__(self, branch1, branch2):
                super().__init__()
                self.branch1 = branch1
                self.branch2 = branch2
            
            def forward(self, x):
                return self.branch1(x) + self.branch2(x)
        
        h_model = ParallelAdd(branch1, branch2)
        
        # Freeze parameters
        for param in h_model.parameters():
            param.requires_grad = False
        
        return h_model
    
    @staticmethod
    def create_closed_loop_model() -> nn.Module:
        """
        Create closed-loop dynamics model: x' = f(x, pi(x))
        
        Returns:
            f_pi_model: Closed-loop dynamics network
        """
        # Create policy
        policy = DoubleIntegrator.create_policy()
        
        # x' = Ax + B*pi(x)
        class ClosedLoopDynamics(nn.Module):
            def __init__(self):
                super().__init__()
                self.policy = policy
                
                # Linear dynamics: Ax
                self.A_layer = nn.Linear(2, 2)
                self.A_layer.weight.data = torch.from_numpy(DoubleIntegrator.A)
                self.A_layer.bias.data = torch.zeros(2)
                
                # Control dynamics: B*u
                self.B_layer = nn.Linear(1, 2)
                self.B_layer.weight.data = torch.from_numpy(DoubleIntegrator.B.reshape(2, 1))
                self.B_layer.bias.data = torch.zeros(2)
                
                # State clipping layers
                # max(x, x_low) = relu(x - x_low) + x_low
                self.clip_low_1 = nn.Linear(2, 2)
                self.clip_low_1.weight.data = torch.eye(2)
                self.clip_low_1.bias.data = torch.from_numpy(-DoubleIntegrator.x_low)
                
                self.clip_low_2 = nn.Linear(2, 2)
                self.clip_low_2.weight.data = torch.eye(2)
                self.clip_low_2.bias.data = torch.from_numpy(DoubleIntegrator.x_low)
                
                # min(x, x_high) = -relu(-x + x_high) + x_high
                self.clip_high_1 = nn.Linear(2, 2)
                self.clip_high_1.weight.data = -torch.eye(2)
                self.clip_high_1.bias.data = torch.from_numpy(DoubleIntegrator.x_high)
                
                self.clip_high_2 = nn.Linear(2, 2)
                self.clip_high_2.weight.data = -torch.eye(2)
                self.clip_high_2.bias.data = torch.from_numpy(DoubleIntegrator.x_high)
            
            def forward(self, x):
                # Compute x' = Ax + B*pi(x)
                x_next = self.A_layer(x) + self.B_layer(self.policy(x))
                
                # Clip to state bounds
                # max(x, x_low)
                x_next = torch.relu(self.clip_low_1(x_next))
                x_next = self.clip_low_2(x_next)
                
                # min(x, x_high)
                x_next = torch.relu(self.clip_high_1(x_next))
                x_next = self.clip_high_2(x_next)
                
                return x_next
        
        f_pi_model = ClosedLoopDynamics()
        
        # Freeze parameters
        for param in f_pi_model.parameters():
            param.requires_grad = False
        
        return f_pi_model


# Create models for convenience
pi_model = DoubleIntegrator.create_policy()
h_model = DoubleIntegrator.create_constraint_model()
f_pi_model = DoubleIntegrator.create_closed_loop_model()
