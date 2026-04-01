import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are


class RobotArm:
    """
    RobotArm benchmark

    State:
        x = [theta_1, ..., theta_n, theta_dot_1, ..., theta_dot_n]

    Action:
        u = [u_1, ..., u_n]

    Constraint:
        h(x) = x_end - x_con
    where x_end is the x-coordinate of the end effector.

    Safe set convention:
        h(x) <= 0
    """

    n = 3
    x_dim = 2 * n
    u_dim = n

    x_low = np.array(
        [np.pi / 3] + [-np.pi / 6] * (n - 1) + [-1.0] * n,
        dtype=np.float32
    )
    x_high = np.array(
        [np.pi / 2] + [np.pi / 6] * (n - 1) + [1.0] * n,
        dtype=np.float32
    )

    u_low = np.array([-1.0] * u_dim, dtype=np.float32)
    u_high = np.array([1.0] * u_dim, dtype=np.float32)

    dt = 0.1

    @staticmethod
    def create_linear_dynamics_matrices():
        n = RobotArm.n
        x_dim = RobotArm.x_dim
        u_dim = RobotArm.u_dim
        dt = RobotArm.dt

        A = np.eye(x_dim, dtype=np.float32)
        for i in range(n):
            A[i, n + i] = dt

        B = np.zeros((x_dim, u_dim), dtype=np.float32)
        for i in range(n):
            B[n + i, i] = dt

        return A, B

    @staticmethod
    def constraint(x: np.ndarray) -> np.ndarray:
        """
        x: [x_dim, N]
        returns: [1, N]
        """
        n = RobotArm.n
        th = np.cumsum(x[:n, :], axis=0)
        x_end = np.sum(np.cos(th), axis=0)
        x_con = 0.5 * n
        h = x_end - x_con
        return h.reshape(1, -1).astype(np.float32)

    @staticmethod
    def constraint_torch(x: torch.Tensor) -> torch.Tensor:
        """
        Accepts either:
          - [batch, x_dim]
          - [x_dim, batch]

        Returns:
          - [batch, 1] if input was [batch, x_dim]
          - [1, batch] if input was [x_dim, batch]
        """
        transposed = False

        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")

        if x.shape[0] == RobotArm.x_dim and x.shape[1] != RobotArm.x_dim:
            x = x.T
            transposed = True
        elif x.shape[1] != RobotArm.x_dim:
            raise ValueError(
                f"Input shape {tuple(x.shape)} is incompatible with x_dim={RobotArm.x_dim}"
            )

        n = RobotArm.n
        th = torch.cumsum(x[:, :n], dim=1)
        x_end = torch.sum(torch.cos(th), dim=1, keepdim=True)
        x_con = 0.5 * n
        h = x_end - x_con

        if transposed:
            h = h.T

        return h

    @staticmethod
    def terminated(x: np.ndarray) -> np.ndarray:
        low_hit = np.any(x <= RobotArm.x_low.reshape(-1, 1), axis=0)
        high_hit = np.any(x >= RobotArm.x_high.reshape(-1, 1), axis=0)
        return low_hit | high_hit

    @staticmethod
    def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Exact one-step closed dynamics:
            x_next = A x + B u
        followed by clipping to state bounds.

        x: [x_dim, N]
        u: [u_dim, N]
        """
        A, B = RobotArm.create_linear_dynamics_matrices()
        u = np.clip(
            u.astype(np.float32),
            RobotArm.u_low.reshape(-1, 1),
            RobotArm.u_high.reshape(-1, 1),
        )
        x_next = (A @ x + B @ u).astype(np.float32)
        x_next = np.clip(
            x_next,
            RobotArm.x_low.reshape(-1, 1),
            RobotArm.x_high.reshape(-1, 1),
        )
        return x_next

    @staticmethod
    def step_torch(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Accepts either:
          - x: [batch, x_dim], u: [batch, u_dim]
          - x: [x_dim, batch], u: [u_dim, batch]

        Returns in the same layout as x.
        """
        transposed = False

        if x.ndim != 2 or u.ndim != 2:
            raise ValueError(
                f"Expected 2D tensors, got x shape {tuple(x.shape)}, u shape {tuple(u.shape)}"
            )

        if x.shape[0] == RobotArm.x_dim and x.shape[1] != RobotArm.x_dim:
            x = x.T
            transposed = True
        elif x.shape[1] != RobotArm.x_dim:
            raise ValueError(
                f"x shape {tuple(x.shape)} is incompatible with x_dim={RobotArm.x_dim}"
            )

        if u.shape[0] == RobotArm.u_dim and u.shape[1] != RobotArm.u_dim:
            u = u.T
        elif u.shape[1] != RobotArm.u_dim:
            raise ValueError(
                f"u shape {tuple(u.shape)} is incompatible with u_dim={RobotArm.u_dim}"
            )

        A_np, B_np = RobotArm.create_linear_dynamics_matrices()
        A = torch.as_tensor(A_np, dtype=x.dtype, device=x.device)
        B = torch.as_tensor(B_np, dtype=x.dtype, device=x.device)
        x_low = torch.as_tensor(RobotArm.x_low, dtype=x.dtype, device=x.device)
        x_high = torch.as_tensor(RobotArm.x_high, dtype=x.dtype, device=x.device)
        u_low = torch.as_tensor(RobotArm.u_low, dtype=x.dtype, device=x.device)
        u_high = torch.as_tensor(RobotArm.u_high, dtype=x.dtype, device=x.device)

        u = torch.maximum(u, u_low)
        u = torch.minimum(u, u_high)

        x_next = x @ A.T + u @ B.T
        x_next = torch.maximum(x_next, x_low)
        x_next = torch.minimum(x_next, x_high)

        if transposed:
            x_next = x_next.T

        return x_next

    @staticmethod
    def create_policy() -> nn.Module:
        n = RobotArm.n
        x_dim = RobotArm.x_dim
        u_dim = RobotArm.u_dim

        A, B = RobotArm.create_linear_dynamics_matrices()

        Q = np.diag([1.0] * n + [0.1] * n).astype(np.float32)
        R = np.diag([0.1] * u_dim).astype(np.float32)

        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        x_ref = np.zeros((x_dim,), dtype=np.float32)
        x_ref[0] = np.pi / 2
        bias = K @ x_ref

        layers = []

        # Dense(-K, K*x_ref)
        linear = nn.Linear(x_dim, u_dim)
        linear.weight.data = torch.from_numpy((-K).astype(np.float32))
        linear.bias.data = torch.from_numpy(bias.astype(np.float32))
        layers.append(linear)

        # max(u, u_low) = relu(u - u_low) + u_low
        l1 = nn.Linear(u_dim, u_dim)
        l1.weight.data = torch.eye(u_dim, dtype=torch.float32)
        l1.bias.data = torch.from_numpy((-RobotArm.u_low).astype(np.float32))
        layers.append(l1)
        layers.append(nn.ReLU())

        l2 = nn.Linear(u_dim, u_dim)
        l2.weight.data = torch.eye(u_dim, dtype=torch.float32)
        l2.bias.data = torch.from_numpy(RobotArm.u_low.astype(np.float32))
        layers.append(l2)

        # min(u, u_high) = -relu(-u + u_high) + u_high
        l3 = nn.Linear(u_dim, u_dim)
        l3.weight.data = -torch.eye(u_dim, dtype=torch.float32)
        l3.bias.data = torch.from_numpy(RobotArm.u_high.astype(np.float32))
        layers.append(l3)
        layers.append(nn.ReLU())

        l4 = nn.Linear(u_dim, u_dim)
        l4.weight.data = -torch.eye(u_dim, dtype=torch.float32)
        l4.bias.data = torch.from_numpy(RobotArm.u_high.astype(np.float32))
        layers.append(l4)

        pi_model = nn.Sequential(*layers)

        for param in pi_model.parameters():
            param.requires_grad = False

        return pi_model

    @staticmethod
    def create_closed_loop_dynamics_model() -> nn.Module:
        A_np, B_np = RobotArm.create_linear_dynamics_matrices()
        x_dim = RobotArm.x_dim
        u_dim = RobotArm.u_dim

        A = A_np.astype(np.float32)
        B = B_np.astype(np.float32)
        I = np.eye(x_dim, dtype=np.float32)

        # dx = (A - I)x + B u, so x_next = x + dx
        D = np.concatenate([A - I, B], axis=1).astype(np.float32)

        pi_model = RobotArm.create_policy()

        f_linear = nn.Linear(x_dim + u_dim, x_dim)
        f_linear.weight.data = torch.from_numpy(D)
        f_linear.bias.data = torch.zeros(x_dim, dtype=torch.float32)
        f_model = nn.Sequential(f_linear)

        for param in f_model.parameters():
            param.requires_grad = False

        class ClosedLoopDynamics(nn.Module):
            def __init__(self):
                super().__init__()
                self.pi_model = pi_model
                self.f_model = f_model

                self.x_dim = RobotArm.x_dim
                self.u_dim = RobotArm.u_dim

                self.register_buffer("A", torch.tensor(A_np, dtype=torch.float32))
                self.register_buffer("B", torch.tensor(B_np, dtype=torch.float32))
                self.register_buffer("x_low", torch.tensor(RobotArm.x_low, dtype=torch.float32))
                self.register_buffer("x_high", torch.tensor(RobotArm.x_high, dtype=torch.float32))

                # identity normalization for MILP compatibility
                self.register_buffer("x_mean", torch.zeros(RobotArm.x_dim, dtype=torch.float32))
                self.register_buffer("x_std", torch.ones(RobotArm.x_dim, dtype=torch.float32))
                self.register_buffer("u_mean", torch.zeros(RobotArm.u_dim, dtype=torch.float32))
                self.register_buffer("u_std", torch.ones(RobotArm.u_dim, dtype=torch.float32))
                self.register_buffer("dx_mean", torch.zeros(RobotArm.x_dim, dtype=torch.float32))
                self.register_buffer("dx_std", torch.ones(RobotArm.x_dim, dtype=torch.float32))

                for param in self.pi_model.parameters():
                    param.requires_grad = False

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Accepts either:
                  - [batch, x_dim]
                  - [x_dim, batch]

                Returns in the same layout as input.
                """
                transposed = False

                if x.ndim != 2:
                    raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")

                if x.shape[0] == RobotArm.x_dim and x.shape[1] != RobotArm.x_dim:
                    x = x.T
                    transposed = True
                elif x.shape[1] != RobotArm.x_dim:
                    raise ValueError(
                        f"Input shape {tuple(x.shape)} is incompatible with x_dim={RobotArm.x_dim}"
                    )

                u = self.pi_model(x)
                xu = torch.cat([x, u], dim=1)
                dx = self.f_model(xu)
                x_next = x + dx
                x_next = torch.maximum(x_next, self.x_low)
                x_next = torch.minimum(x_next, self.x_high)

                if transposed:
                    x_next = x_next.T

                return x_next

        model = ClosedLoopDynamics()

        for param in model.parameters():
            param.requires_grad = False

        return model


# aliases for convenience
pi_model = RobotArm.create_policy()
f_pi_model = RobotArm.create_closed_loop_dynamics_model()