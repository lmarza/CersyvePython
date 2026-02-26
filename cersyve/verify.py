from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# alpha-beta-CROWN "new API" objects:
#   - VerificationSpec encodes the input set X and the unsafe output set (DNF constraints)
#   - ABCrownSolver runs attack (PGD) and/or complete verification (BaB) depending on config
from abcrown import ABCrownSolver, VerificationSpec, ConfigBuilder, input_vars, output_vars
import torch
import torch.nn as nn

import torch
import torch.nn as nn

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

class ABCrownAdapter(nn.Module):
    """
    We verify neural nets over an input hyper-rectangle X and look for a counterexample x in X.
    CersyvePython models expect column-major batches:
         input  x has shape [x_dim, N]
         output y has shape [k,     N]

    alpha-beta-CROWN expects the standard PyTorch batch-first convention:
         input  x has shape [N, x_dim]
         output y has shape [N, k]

    This adapter is a tensor-shape bridge; it does not change the function being verified.
    """
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, x_dim] (abcrown convention)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, x_dim]
        # convert to [x_dim, N]
        x_t = x.transpose(0, 1).contiguous()
        # core outputs [2, N] (for the property NN used in verification)
        y = self.core(x_t)
        # convert back to [N, 2] (abcrown convention)
        y = y.transpose(0, 1).contiguous()
        return y


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
    v_thresh: float = 0.0,
    other_thresh: float = 0.0,
    strict_margin: float = 1e-6,
    device: str = "cpu",
    timeout: Optional[float] = None,
    bab: bool = True,
) -> VerifyResult:
    """
    Verification is turned into a counterexample search problem: "Does there exist x in X such that the output y lies in the unsafe region?"

    Here:
      - X is a box:  x in [x_low, x_high]
      - The network output is 2D: y = [y0, y1]
      - Unsafe region is the "second quadrant" used in the paper:
            y0 <= 0   and   y1 >= 0
        where y0 is V(x), and y1 is either h(x) or V(f_pi(x)) depending on the property.

    alpha-beta-CROWN spec requires strict comparisons (<, >), so we implement a slightly
    shrunken unsafe region:
            y0 <  0 - strict_margin   and   y1 > 0 + strict_margin
    """
    # validate box bounds
    x_low = np.asarray(x_low, dtype=np.float32)
    x_high = np.asarray(x_high, dtype=np.float32)
    assert x_low.shape == x_high.shape
    x_dim = x_low.size

    # Wrap the property network in an adapter so abcrown sees [N,x_dim] -> [N,2]
    # graph = ABCrownAdapter(model_2out).to(device)
    graph = model_2out.to(device)

    # Create symbolic variables used by the spec
    # x is the input variable (dimension x_dim)
    # y is the output variable (dimension 2)
    x = input_vars((x_dim,))
    y = output_vars(2)

    # Input set X: closed hyper-rectangle
    low_t = torch.tensor(x_low, device=device)
    high_t = torch.tensor(x_high, device=device)
    input_constraint = (x >= low_t) & (x <= high_t)

    output_constraint = (y[0] > v_thresh) | (y[1] < other_thresh)
    # output_constraint = (y[0] <= v_thresh) & (y[1] <= other_thresh)

    # Build the full verification problem: "exists x in X s.t. y(x) in unsafe set?"
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )

    # Configure solver:
    #   - device: cpu/cuda
    #   - complete_verifier: "bab" means branch-and-bound is available for complete proofs
    cfg = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(complete_verifier__complete_verifier=("bab" if bab else "auto"))
    )()

    # Optional global timeout for the verification run
    if timeout is not None:
        cfg = ConfigBuilder.from_config(cfg).update({"general": {"timeout": float(timeout)}})()

    # Run verification abcrown:
    #   1) attack (PGD) tries to find a concrete counterexample fast -> "unsafe-pgd"
    #   2) if needed, BaB tries to prove UNSAT -> "verified"/"safe", or find "unsafe-bab"
    solver = ABCrownSolver(spec, graph, config=cfg)
    res = solver.solve()

    # Extract an x* (counterexample) if one was found
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
    strict_margin: float = 1e-6,
    device: str = "cpu",
    timeout_s: Optional[float] = 300.0,
) -> Tuple[VerifyResult, VerifyResult]:
    """
    Verifies two properties by checking emptiness of two unsafe sets:
    (1) Constraint property network:
        y(x) = [V(x), h(x)]
        unsafe if V(x) <= 0 and h(x) >= 0
        ("V says safe, but constraint says unsafe")
    (2) Invariance property network:
        y(x) = [V(x), V(f_pi(x))]
        unsafe if V(x) <= 0 and V(f_pi(x)) >= 0
        ("V decreases condition violated over one closed-loop step")

    We implement the same checks with strict inequalities using strict_margin.
    """
    #
    # with torch.no_grad():
    #     xcol = torch.tensor([[-1.0], [-1.0]], device=device)  # [2,1]
    #     y_con = V_h_model(xcol).detach().cpu().numpy().reshape(-1)  # should be length 2
    #     y_inv = V_V_prime_model(xcol).detach().cpu().numpy().reshape(-1)
    # print("V_h_model([-1,-1]) =", y_con, " (interpreted as [?, ?])")
    # print("V_Vp_model([-1,-1]) =", y_inv, " (interpreted as [?, ?])")

    # (A) Constraint check: exists x s.t. V(x) < 0 and h(x) > 0 ?
    con = verify_property(
        V_h_model,
        x_low,
        x_high,
        v_thresh=0.0,
        other_thresh=0.0,
        strict_margin=strict_margin,
        device=device,
        timeout=timeout_s,
        bab=True,
    )

    # (B) Invariance check: exists x s.t. V(x) < 0 and V(f(x)) > 0 ?
    inv = verify_property(
        V_V_prime_model,
        x_low,
        x_high,
        v_thresh=0.0,
        other_thresh=0.0,
        strict_margin=strict_margin,
        device=device,
        timeout=timeout_s,
        bab=True,
    )

    return con, inv