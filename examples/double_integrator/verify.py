import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn as nn

from systems.double_integrator import DoubleIntegrator
from cersyve import create_mlp, create_closed_loop_dynamics_model
from cersyve.verify import (
    verify_value,
    ClosedLoopDynamicsBatchFirst,
    ValueConstraintPropertyBF,
    ValueNextValuePropertyBF,
)


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

    # Build batch-first property networks for ab-crown
    f_pi_bf = ClosedLoopDynamicsBatchFirst(
        f_model, pi_model, data,
        DoubleIntegrator.x_low, DoubleIntegrator.x_high,
    ).to(device).eval()

    V_h_model = ValueConstraintPropertyBF(V_model, h_model).to(device).eval()
    V_V_prime_model = ValueNextValuePropertyBF(V_model, f_pi_bf).to(device).eval()

    # Run verifier
    con_res, inv_res = verify_value(
        DoubleIntegrator.x_low,
        DoubleIntegrator.x_high,
        V_h_model,
        V_V_prime_model,
        device=device,
        timeout_s=300.0,
    )

    print("\n=== Verification results (alpha-beta-CROWN) ===")
    print(f"Constraint property: {con_res.status} (success={con_res.success})")
    if con_res.status == 'safe' and con_res.success:
        print("  No violation exists: property holds everywhere in the input domain.")
    if con_res.counterexample is not None:
        print(f"  Closest input to violation x*: {con_res.counterexample}")
        x_star = torch.tensor(con_res.counterexample, dtype=torch.float32, device=device).view(1, -1)
        with torch.no_grad():
            Vx = V_model(x_star).item()
            hx = h_model(x_star).item()
        print(f"  V(x*): {Vx:.6f}  (must be <= 0 for a true violation)")
        print(f"  h(x*): {hx:.6f}  (must be > 0 for a true violation)")
        if Vx > 0:
            print(f"  *** SPURIOUS: V(x*) > 0 means x* is outside the safe set ***")

    print(f"\nInvariance property: {inv_res.status} (success={inv_res.success})")
    if inv_res.status == 'safe' and inv_res.success:
        print("  No violation exists: property holds everywhere in the input domain.")
    if inv_res.counterexample is not None:
        print(f"  Closest input to violation x*: {inv_res.counterexample}")
        x_star = torch.tensor(inv_res.counterexample, dtype=torch.float32, device=device).view(1, -1)
        with torch.no_grad():
            Vx = V_model(x_star).item()
            x_next = f_pi_bf(x_star)
            Vnext = V_model(x_next).item()
        print(f"  V(x*): {Vx:.6f}  (must be <= 0 for a true violation)")
        print(f"  V(f_pi(x*)): {Vnext:.6f}  (must be > 0 for a true violation)")
        if Vx > 0:
            print(f"  *** SPURIOUS: V(x*) > 0 means x* is outside the safe set ***")


if __name__ == "__main__":
    main()