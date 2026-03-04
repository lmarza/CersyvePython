from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# alpha-beta-CROWN "new API" objects:
#   - VerificationSpec encodes the input set X and the safe output property
#     (the library internally negates it to get the unsafe condition)
#   - ABCrownSolver runs attack (PGD) and/or complete verification (BaB) depending on config
from abcrown import ABCrownSolver, VerificationSpec, ConfigBuilder, input_vars, output_vars

from cersyve.model import ClosedLoopDynamicsModel


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
        x_norm = (x - self.x_mean.view(1, -1)) / self.x_std.view(1, -1)

        u = self.pi_model(x)  # [N, u_dim]
        u_norm = (u - self.u_mean.view(1, -1)) / self.u_std.view(1, -1)

        xu_norm = torch.cat([x_norm, u_norm], dim=1)  # concat axis=1

        dx = self.f_model(xu_norm) * self.dx_std.view(1, -1) + self.dx_mean.view(1, -1)
        x_next = x + dx

        # clip with relu trick, batch-first
        x_next = torch.relu(x_next - self.x_low.view(1, -1)) + self.x_low.view(1, -1)
        x_next = -torch.relu(-x_next + self.x_high.view(1, -1)) + self.x_high.view(1, -1)
        return x_next

class ClosedLoopDynamicsBatchFirstWrapper(nn.Module):
    """
    Wrapper to adapt a row-major ClosedLoopDynamicsModel (x: [x_dim, N]) to batch-first
    semantics (x: [N, x_dim]) by transposing inputs/outputs.
    """
    def __init__(self, f_pi_model: ClosedLoopDynamicsModel):
        super().__init__()
        self.f_pi_model = f_pi_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, x_dim] batch-first
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Handle single-sample case: [1, x_dim]

        x_t = x.T.contiguous()  # Transpose to row-major: [x_dim, N]
        x_next_t = self.f_pi_model(x_t)  # Call the original forward: [x_dim, N]
        x_next = x_next_t.T.contiguous()  # Transpose back to batch-first: [N, x_dim]
        return x_next

class ValueConstraintPropertyBF(nn.Module):
    """y=[V(x), h(x)] shaped [N,2]."""
    def __init__(self, V_model, h_model):
        super().__init__()
        self.V = V_model
        self.h = h_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Vx = self.V(x)  # [N,1]
        hx = self.h(x)  # [N,1]
        return torch.cat([Vx, hx], dim=1)  # [N,2]

class ValueNextValuePropertyBF(nn.Module):
    """y=[V(x), V(f_pi(x))] shaped [N,2]."""
    def __init__(self, V_model, f_pi_bf):
        super().__init__()
        self.V = V_model
        self.f_pi = f_pi_bf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Vx = self.V(x)           # [N,1]
        x_next = self.f_pi(x)    # [N,x_dim]
        Vnext = self.V(x_next)   # [N,1]
        return torch.cat([Vx, Vnext], dim=1)  # [N,2]


@dataclass
class VerifyResult:
    """
    Mirrors the high-level outcomes in the paper:
      - holds/verified: UNSAT (no counterexample exists in X)
      - violated/unsafe: SAT (a concrete counterexample x* exists)
      - unknown: neither proven nor refuted within limits
    """
    status: str                 # e.g. "verified", "unsafe-pgd", "unsafe-bab", "unknown"
    success: bool
    counterexample: Optional[np.ndarray] = None  # x* in R^{x_dim} if found
    raw: Optional[Dict[str, Any]] = None         # debug info


def _extract_counterexample(res) -> Optional[np.ndarray]:
    """
    alpha-beta-CROWN can return counterexamples from different stages.
    SolveResult.reference is empty for PGD, and the concrete adversarial inputs are stored in:
        res.stats['attack_examples']   (Tensor of shape [K, x_dim])
    We return the first example as a numpy array [x_dim].
    """
    stats = getattr(res, "stats", None) or {}

    x = stats.get("attack_examples", None)
    if x is None:
        x = stats.get("all_adv_candidates", None)
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dim() == 1:
            return x.numpy().astype(np.float32)
        if x.dim() == 2 and x.shape[0] >= 1:
            return x[0].numpy().astype(np.float32)

    # last-resort coercion
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[0] >= 1:
        return x[0]
    return None


def verify_property(
    model_2out: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    *,
    device: str = "cpu",
    timeout: Optional[float] = None,
    bab: bool = True,
) -> VerifyResult:
    """
    Verify a two-output property network over input box [x_low, x_high].

    The network outputs y = [y0, y1] where:
      - For constraint check:  y0 = V(x),  y1 = h(x)
      - For invariance check:  y0 = V(x),  y1 = V(f_pi(x))

    The **safe property** (must hold for ALL x) is:
        y0 > 0  OR  y1 < 0
    i.e. either x is outside the safe set (V > 0) or the second output is
    non-positive (h < 0 for constraint, V_next < 0 for invariance).

    IMPORTANT:  ``output_constraint`` in the ab-crown API defines the
    **safe property** that the solver tries to prove for all inputs.
    The library internally NEGATES it (``negate=True`` in
    ``_predicate_to_dnf``) to derive the conjunctive unsafe condition
        y0 <= 0  AND  y1 >= 0
    which the solver searches for.  A disjunctive safe property
    produces the correct single conjunctive clause after negation.
    """
    x_low = np.asarray(x_low, dtype=np.float32)
    x_high = np.asarray(x_high, dtype=np.float32)
    assert x_low.shape == x_high.shape
    x_dim = x_low.size

    graph = model_2out.to(device)

    # Create symbolic variables used by the spec
    x = input_vars((x_dim,))
    y = output_vars(2)

    # Input set X: closed hyper-rectangle
    low_t = torch.tensor(x_low, device=device)
    high_t = torch.tensor(x_high, device=device)
    input_constraint = (x >= low_t) & (x <= high_t)

    # Safe property (disjunctive):  V(x) > 0  OR  second_output < 0
    # ab-crown internally negates this to the conjunctive unsafe condition:
    #   V(x) <= 0  AND  second_output >= 0
    # and searches for an x satisfying that conjunction.
    output_constraint = (y[0] > 0) | (y[1] < 0)

    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )

    cfg = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(complete_verifier__complete_verifier=("bab" if bab else "auto"))
    )()

    if timeout is not None:
        cfg = ConfigBuilder.from_config(cfg).update({"general": {"timeout": float(timeout)}})()

    solver = ABCrownSolver(spec, graph, config=cfg)
    res = solver.solve()

    ce = _extract_counterexample(res)

    return VerifyResult(
        status=str(getattr(res, "status", "unknown")),
        success=bool(getattr(res, "success", False)),
        counterexample=ce,
        raw=getattr(res, "stats", None),
    )


def verify_value(
    x_low: np.ndarray,
    x_high: np.ndarray,
    V_h_model: nn.Module,
    V_V_prime_model: nn.Module,
    *,
    device: str = "cpu",
    timeout_s: Optional[float] = 300.0,
) -> Tuple[VerifyResult, VerifyResult]:
    """
    Verifies two properties using a disjunctive safe property.

    (1) Constraint property network:  y = [V(x), h(x)]
        Safe property:  V(x) > 0  OR  h(x) < 0
        ("x is outside the safe set, or constraint is satisfied")
        After negation -> solver searches for:  V(x) <= 0  AND  h(x) >= 0

    (2) Invariance property network:  y = [V(x), V(f_pi(x))]
        Safe property:  V(x) > 0  OR  V(f_pi(x)) < 0
        ("x is outside the safe set, or next-step value is non-positive")
        After negation -> solver searches for:  V(x) <= 0  AND  V(f_pi(x)) >= 0

    Both properties are encoded as a disjunctive output_constraint
    which the ab-crown library internally negates into a single
    conjunctive unsafe clause.
    """
    # (A) Constraint check:  exists x s.t. V(x) <= 0 AND h(x) > 0 ?
    con = verify_property(
        V_h_model,
        x_low,
        x_high,
        device=device,
        timeout=timeout_s,
        bab=True,
    )

    # (B) Invariance check:  exists x s.t. V(x) <= 0 AND V(f_pi(x)) > 0 ?
    inv = verify_property(
        V_V_prime_model,
        x_low,
        x_high,
        device=device,
        timeout=timeout_s,
        bab=True,
    )

    return con, inv