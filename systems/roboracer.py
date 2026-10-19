"""Obstacle-agnostic tube certificate for the lab track.

x = [e_y, e_psi, y_left, y_right], one step per ds metres of travel (arclength, so the
certificate is speed-free). e_y, e_psi are the car's offset and heading error relative to
a planned path; y_left, y_right are the free room for the car's CENTER on each side of
that path (walls and inflated obstacles minus the car's corner extent), held constant.

    h(x) = max(e_y - y_left, -y_right - e_y, |e_psi| - psi_con)

Control is a curvature correction u on top of an exact runtime feedforward. With car
arclength sigma, de_psi/dsigma = kappa_cmd - kappa cos(e_psi) / (1 - kappa e_y); the
runtime commands kappa_cmd = u + kappa cos(e_psi) / (1 - kappa e_y), which makes
de_psi/dsigma = u exactly for any path curvature kappa. The steering authority left for
u is tan(delta_max)/L - KAPPA_FF_MAX, and the runtime brakes whenever the feedforward
would exceed KAPPA_FF_MAX. The one modelling approximation left is sin(e_psi) ~ e_psi in
de_y/dsigma, bounded by |e_psi| <= psi_con.
Steering delay and braking are handled at runtime, outside the certificate.
"""
import numpy as np
import torch
import torch.nn as nn

from scipy.linalg import solve_discrete_are

WHEELBASE = 0.3275
DELTA_MAX = 0.3490658503988659
KAPPA_FF_MAX = 0.80                                  # runtime brakes above this feedforward
U_MAX = float(np.tan(DELTA_MAX) / WHEELBASE - KAPPA_FF_MAX)   # ~0.31 1/m



def _lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K.astype(np.float32)


class RoboRacer:
    x_dim = 4
    u_dim = 1
    ds = 0.05
    psi_con = 0.40
    x_low = np.array([-0.70, -0.60, 0.05, 0.05], dtype=np.float32)
    x_high = np.array([0.70, 0.60, 0.60, 0.60], dtype=np.float32)
    u_low = np.array([-U_MAX], dtype=np.float32)
    u_high = np.array([U_MAX], dtype=np.float32)
    wheelbase = WHEELBASE
    kappa_ff_max = KAPPA_FF_MAX

    # exact integration of e_y'' = u over one ds step
    A = np.eye(4, dtype=np.float32)
    A[0, 1] = ds
    B = np.zeros((4, 1), dtype=np.float32)
    B[0, 0] = 0.5 * ds * ds
    B[1, 0] = ds

    _K2 = _lqr_gain(A[:2, :2].astype(np.float64), B[:2].astype(np.float64),
                    np.diag([4.0, 1.0]), np.array([[0.5]]))

    e_cap = 0.25    # offset the backup reacts to; caps the return heading near 0.25 rad

    @classmethod
    def create_policy(cls):
        """u = clip(-k0 * clip(e_y - (y_left - y_right)/2, +-e_cap) - k1 * e_psi, u bounds).

        Capping the offset caps the heading the car returns with, so a far-off car does
        not steer past psi_con on its way back.
        """
        k0, k1 = float(cls._K2[0, 0]), float(cls._K2[0, 1])
        d, ul, uh = cls.e_cap, float(cls.u_low[0]), float(cls.u_high[0])

        def lin(W, b):
            l = nn.Linear(len(W[0]), len(W))
            l.weight.data = torch.tensor(W, dtype=torch.float32)
            l.bias.data = torch.tensor(b, dtype=torch.float32)
            return l
        # z = e_y - yl/2 + yr/2 ; channels relu(z + d), relu(z - d), relu(e_psi), relu(-e_psi)
        L1 = lin([[1, 0, -0.5, 0.5], [1, 0, -0.5, 0.5], [0, 1, 0, 0], [0, -1, 0, 0]],
                 [d, -d, 0, 0])
        # clip(z) = relu(z+d) - relu(z-d) - d ; r = u - ul
        L2 = lin([[-k0, k0, -k1, k1]], [k0 * d - ul])
        L3 = lin([[-1.0]], [uh - ul])          # relu(uh - max(u, ul))
        L4 = lin([[-1.0]], [uh])               # min(max(u, ul), uh)
        model = nn.Sequential(L1, nn.ReLU(), L2, nn.ReLU(), L3, nn.ReLU(), L4)
        for prm in model.parameters():
            prm.requires_grad = False
        return model

    @classmethod
    def dynamics(cls, x, u):
        """Rollout model (columns are states), identical to the certified linear map."""
        u = np.clip(u[0, :], cls.u_low[0], cls.u_high[0])
        x_next = np.stack([
            x[0] + cls.ds * x[1] + 0.5 * cls.ds * cls.ds * u,
            x[1] + cls.ds * u,
            x[2],
            x[3],
        ], axis=0).astype(np.float32)
        return np.clip(x_next, cls.x_low.reshape(-1, 1), cls.x_high.reshape(-1, 1))

    @classmethod
    def constraint(cls, x):
        h = np.maximum(np.maximum(x[0] - x[2], -x[3] - x[0]), np.abs(x[1]) - cls.psi_con)
        return h.reshape(1, -1)

    @classmethod
    def terminated(cls, x):
        return (np.isclose(x[0], cls.x_low[0]) | np.isclose(x[0], cls.x_high[0]) |
                np.isclose(x[1], cls.x_low[1]) | np.isclose(x[1], cls.x_high[1]))

    @classmethod
    def create_constraint_model(cls):
        """Exact Linear/ReLU h = max(a, b, c), a = e_y - y_left, b = -y_right - e_y,
        c = |e_psi| - psi_con, via |z| = relu(z) + relu(-z) and max(p, q) = p + relu(q - p)."""
        def lin(W, b):
            W = np.asarray(W, dtype=np.float32)
            l = nn.Linear(W.shape[1], W.shape[0])
            l.weight.data = torch.tensor(W, dtype=torch.float32)
            l.bias.data = torch.tensor(b, dtype=torch.float32)
            return l
        #          e_y  e_psi  yl   yr
        W1 = [[ 1,  0, -1,  0],   # a
              [-1,  0,  1,  0],   # -a
              [-1,  0,  0, -1],   # b
              [ 1,  0,  0,  1],   # -b
              [ 0,  1,  0,  0],   # e_psi
              [ 0, -1,  0,  0]]   # -e_psi
        # relu -> [a+, a-, b+, b-, p+, p-]
        W2 = [[ 1, -1,  0,  0, 0, 0],   # a
              [-1,  1,  0,  0, 0, 0],   # -a
              [-1,  1,  1, -1, 0, 0],   # b - a
              [ 0,  0,  0,  0, 1, 1],   # c = |e_psi| - psi_con
              [ 0,  0,  0,  0,-1,-1]]   # -c
        b2 = [0, 0, 0, -cls.psi_con, cls.psi_con]
        # relu -> [a+, a-, relu(b-a), c+, c-];  m1 = max(a, b) = a + relu(b - a)
        W3 = [[ 1, -1,  1, 0,  0],     # m1
              [-1,  1, -1, 0,  0],     # -m1
              [-1,  1, -1, 1, -1]]     # c - m1
        # relu -> [m1+, m1-, relu(c - m1)];  h = m1 + relu(c - m1)
        W4 = [[1, -1, 1]]
        model = nn.Sequential(lin(W1, [0]*6), nn.ReLU(), lin(W2, b2), nn.ReLU(),
                              lin(W3, [0]*3), nn.ReLU(), lin(W4, [0]))
        for prm in model.parameters():
            prm.requires_grad = False
        return model

    @classmethod
    def create_closed_loop_dynamics_model(cls):
        from cersyve.model import create_linear_closed_loop_model
        return create_linear_closed_loop_model(
            cls.A, cls.B, cls.create_policy(), cls.x_low, cls.x_high)
