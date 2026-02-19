# cersyve/verify_abcrown.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# alpha-beta-CROWN new API
from abcrown import ABCrownSolver, VerificationSpec, ConfigBuilder, input_vars, output_vars


class ABCrownAdapter(nn.Module):
    """
    Adapts Cersyve models that expect x shaped [x_dim, N] and output [2, N]
    into an nn.Module that accepts [N, x_dim] and outputs [N, 2].
    """
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, x_dim] (abcrown convention)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, x_dim]
        x_t = x.transpose(0, 1).contiguous()      # [x_dim, N]
        y = self.core(x_t)                        # [2, N]
        y = y.transpose(0, 1).contiguous()        # [N, 2]
        return y


@dataclass
class VerifyResult:
    status: str                 # e.g. "verified", "unsafe-pgd", "unsafe-bab", "unknown"
    success: bool
    counterexample: Optional[np.ndarray] = None  # shape [x_dim], if available
    raw: Optional[Dict[str, Any]] = None         # solver.reference for debugging


def _extract_counterexample(reference: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    alpha-beta-CROWN stores different keys depending on path (PGD vs BaB).
    This helper is defensive: it scans common fields.
    """
    if not reference:
        return None

    # Common candidates seen in toolchains; keep it permissive.
    candidate_keys = [
        "counterexample", "ce", "adv_example", "adversarial_example",
        "x", "input", "attack_x"
    ]
    for k in candidate_keys:
        if k in reference:
            x = reference[k]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            # normalize to [x_dim]
            if x.ndim == 2 and x.shape[0] == 1:
                x = x[0]
            return x

    return None


def verify_one_property_abcrown(
    model_2out: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    *,
    # unsafe region: y0 < v_thresh AND y1 > other_thresh
    v_thresh: float = 0.0,
    other_thresh: float = 0.0,
    # abcrown requires strict output constraints; add margin to avoid boundary ambiguity
    strict_margin: float = 1e-6,
    device: str = "cpu",
    timeout: Optional[float] = None,
    bab: bool = True,
) -> VerifyResult:
    """
    Proves: for all x in box [x_low, x_high], NOT( y0 < v_thresh AND y1 > other_thresh )
    i.e., UNSAT of unsafe set, else returns a counterexample x.
    """
    x_low = np.asarray(x_low, dtype=np.float32)
    x_high = np.asarray(x_high, dtype=np.float32)
    assert x_low.shape == x_high.shape
    x_dim = x_low.size

    graph = ABCrownAdapter(model_2out).to(device)

    # Symbolic vars for spec
    x = input_vars((x_dim,))
    y = output_vars(2)

    # Input box constraint (non-strict is allowed for inputs)
    low_t = torch.tensor(x_low, device=device)
    high_t = torch.tensor(x_high, device=device)
    input_constraint = (x >= low_t) & (x <= high_t)

    # Unsafe output constraint (MUST be strict per API docs)
    # unsafe if V(x) < v_thresh  AND second_output > other_thresh
    output_constraint = (y[0] < (v_thresh - strict_margin)) & (y[1] > (other_thresh + strict_margin))

    spec = VerificationSpec.build_spec(
        input_vars=x, output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )

    cfg = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        # Use complete verifier unless you explicitly want incomplete only
        .set(complete_verifier__complete_verifier=("bab" if bab else "auto"))
    )()

    if timeout is not None:
        # abcrown config names can change; keep this optional.
        # If your version supports timeouts, set it here.
        cfg = ConfigBuilder.from_config(cfg).update({"general": {"timeout": float(timeout)}})()

    solver = ABCrownSolver(spec, graph, config=cfg)
    res = solver.solve()

    ce = _extract_counterexample(getattr(res, "reference", None))
    return VerifyResult(
        status=str(getattr(res, "status", "unknown")),
        success=bool(getattr(res, "success", False)),
        counterexample=ce,
        raw=getattr(res, "reference", None),
    )


def verify_value_abcrown(
    x_low: np.ndarray,
    x_high: np.ndarray,
    V_h_model: nn.Module,
    V_V_prime_model: nn.Module,
    *,
    strict_margin: float = 1e-6,
    device: str = "cpu",
    timeout: Optional[float] = None,
) -> Tuple[VerifyResult, VerifyResult]:
    """
    Julia equivalent of verify_value(...) but using alpha-beta-CROWN:
      - constraint: find x s.t. V(x) < 0 and h(x) > 0
      - invariance: find x s.t. V(x) < 0 and V(f(x)) > 0
    """
    con = verify_one_property_abcrown(
        V_h_model, x_low, x_high,
        v_thresh=0.0, other_thresh=0.0,
        strict_margin=strict_margin,
        device=device, timeout=timeout, bab=True
    )
    inv = verify_one_property_abcrown(
        V_V_prime_model, x_low, x_high,
        v_thresh=0.0, other_thresh=0.0,
        strict_margin=strict_margin,
        device=device, timeout=timeout, bab=True
    )
    return con, inv
