import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

from systems.double_integrator import DoubleIntegrator
from cersyve import (
    create_mlp,
    create_closed_loop_dynamics_model,
    create_value_constraint_model,
    create_value_next_value_model,
)
import torch.nn as nn
from cersyve.verify import verify_value, ValueConstraintPropertyBF, \
    ValueNextValuePropertyBF

class ClosedLoopDynamicsBatchFirst(nn.Module):
    """abcrown-friendly closed loop: x:[N,x_dim] -> x_next:[N,x_dim], concat on dim=1."""
    def __init__(self, f_model, pi_model, data, x_low, x_high):
        super().__init__()
        self.f_model = f_model
        self.pi_model = pi_model

        # register buffers as float tensors
        self.register_buffer("x_mean", torch.tensor(data["x_mean"], dtype=torch.float32))
        self.register_buffer("x_std",  torch.tensor(data["x_std"],  dtype=torch.float32))
        self.register_buffer("u_mean", torch.tensor(data["u_mean"], dtype=torch.float32))
        self.register_buffer("u_std",  torch.tensor(data["u_std"],  dtype=torch.float32))
        self.register_buffer("dx_mean", torch.tensor(data["dx_mean"], dtype=torch.float32))
        self.register_buffer("dx_std",  torch.tensor(data["dx_std"],  dtype=torch.float32))

        self.register_buffer("x_low",  torch.tensor(x_low,  dtype=torch.float32))
        self.register_buffer("x_high", torch.tensor(x_high, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, x_dim]
        x_norm = (x - self.x_mean) / self.x_std

        u = self.pi_model(x)  # [N, u_dim]
        u_norm = (u - self.u_mean) / self.u_std

        xu_norm = torch.cat([x_norm, u_norm], dim=1)  # concat axis=1

        dx = self.f_model(xu_norm) * self.dx_std + self.dx_mean
        x_next = x + dx

        # clip with relu trick, batch-first
        x_next = torch.relu(x_next - self.x_low) + self.x_low
        x_next = -torch.relu(-x_next + self.x_high) + self.x_high
        return x_next


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_path = "../../data/double_integrator_data.npz"
    print(f"Loading data from {data_path}...")
    data_npz = np.load(data_path)
    data = {k: data_npz[k] for k in data_npz.keys()}

    # Load dynamics model f(x,u)->dx
    dynamics_hidden_sizes = [32, 32]
    f_model = create_mlp(DoubleIntegrator.x_dim + DoubleIntegrator.u_dim, DoubleIntegrator.x_dim, dynamics_hidden_sizes)
    f_model_path = "../../models/double_integrator/f.pth"
    print(f"Loading f_model from {f_model_path}...")
    f_model.load_state_dict(torch.load(f_model_path, map_location=device, weights_only=True))
    f_model.to(device).eval()

    # Build closed-loop dynamics f_pi
    pi_model = DoubleIntegrator.create_policy().to(device).eval()
    f_pi_model = create_closed_loop_dynamics_model(
        f_model, pi_model, data, DoubleIntegrator.x_low, DoubleIntegrator.x_high, DoubleIntegrator.u_dim
    ).to(device).eval()
    print("Created f_pi_model")

    # Load constraint model h(x)
    constraint_hidden_sizes = [16]
    h_model = create_mlp(DoubleIntegrator.x_dim, 1, constraint_hidden_sizes)
    h_model_path = "../../models/double_integrator/h.pth"
    print(f"Loading h_model from {h_model_path}...")
    h_model.load_state_dict(torch.load(h_model_path, map_location=device, weights_only=True))
    h_model.to(device).eval()

    # Load value model V(x) (use finetuned by default)
    value_hidden_sizes = [32, 32]
    V_model = create_mlp(DoubleIntegrator.x_dim, 1, value_hidden_sizes)
    V_path = "../../models/double_integrator/V_finetune_milp.pth"
    print(f"Loading V_model from {V_path}...")
    V_model.load_state_dict(torch.load(V_path, map_location=device, weights_only=True))
    V_model.to(device).eval()

    # Build property networks
    # V_h_model = create_value_constraint_model(V_model, h_model).to(device).eval()
    # V_V_prime_model = create_value_next_value_model(V_model, f_pi_model).to(device).eval()
    f_pi_bf = ClosedLoopDynamicsBatchFirst(f_model, pi_model, data, DoubleIntegrator.x_low, DoubleIntegrator.x_high).to(
        device).eval()
    V_h_model = ValueConstraintPropertyBF(V_model, h_model).to(device).eval()
    V_V_prime_model = ValueNextValuePropertyBF(V_model, f_pi_bf).to(device).eval()

    # Run verifier
    con_res, inv_res = verify_value(
        DoubleIntegrator.x_low,
        DoubleIntegrator.x_high,
        V_h_model,
        V_V_prime_model,
        strict_margin=1e-6, # for output constraints strict inequalities
        device=device,
        timeout_s=300.0,
    )

    print("\n=== Verification results (alpha-beta-CROWN) ===")
    print(f"Constraint property: {con_res.status} (success={con_res.success})")
    if con_res.counterexample is not None:
        print(f"  Counterexample x*: {con_res.counterexample}")
        x_star = torch.tensor(con_res.counterexample, dtype=torch.float32, device=device).view(1, -1)

        with torch.no_grad():
            Vx = V_model(x_star).item()
            hx = h_model(x_star).item()
            # x_next = f_pi_bf(x_star)
            # Vnext = V_model(x_next).item()

        print(f"  V(x*): {Vx:.6f}")
        print(f"  h(x*): {hx:.6f}")
        # print(f"  V(f_pi(x*)): {Vnext:.6f}")


    print(f"Invariance property: {inv_res.status} (success={inv_res.success})")
    if inv_res.counterexample is not None:
        print(f"  Counterexample x*: {inv_res.counterexample}")

if __name__ == "__main__":
    main()