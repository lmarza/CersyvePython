# MILP verification of V.
#   Part 1: reference verifier, PuLP -> Gurobi (milp_verify)
#   Part 2: direct Gurobi verifier with LP bounds and box splitting (milp_fast_verify)
"""MILP formal verification for ReLU neural network CBFs.

Encodes piecewise-linear (ReLU) neural networks as Mixed-Integer Linear Programs
and solves for counterexamples to constraint satisfaction and forward invariance.

Solver: PuLP front end; Gurobi by default (CERSYVE_MILP_SOLVER or
solver_kwargs["solver"] select another PuLP solver).

Accounting: every public check accepts ``record``, a list that receives one
dict per MILP solve (stage, verdict, status, objective, seconds), so a caller
can reconcile verifier invocations with the actual number of solver runs.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import pulp

from .tolerances import TRAIN_MARGIN, VERIFY_MARGIN


# ─────────────────────────────────────────────────────────────────────────────
# Interval arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def _interval_bound_linear(W, b, lb, ub):
    """Interval arithmetic for y = Wx + b."""
    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)
    y_lb = W_pos @ lb + W_neg @ ub + b
    y_ub = W_pos @ ub + W_neg @ lb + b
    return y_lb, y_ub


# ─────────────────────────────────────────────────────────────────────────────
# MILP building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _milp_add_linear(prob, x_vars, W, b, prefix):
    """Add constraints for y = Wx + b, return y_vars."""
    n_out = W.shape[0]
    y_vars = []
    for j in range(n_out):
        y = pulp.LpVariable(f"{prefix}_{j}")
        prob += (
            y == pulp.lpSum(float(W[j, k]) * x_vars[k] for k in range(len(x_vars)))
            + float(b[j])
        )
        y_vars.append(y)
    return y_vars


def _milp_add_relu(prob, z_vars, z_lb, z_ub, prefix):
    """Add ReLU constraints y = max(0, z) using big-M formulation."""
    y_vars = []
    for j in range(len(z_vars)):
        L, U = float(z_lb[j]), float(z_ub[j])
        if L >= 0:
            # ReLU is identity
            y = pulp.LpVariable(f"{prefix}_{j}", lowBound=L, upBound=U)
            prob += y == z_vars[j]
        elif U <= 0:
            # ReLU is zero
            y = pulp.LpVariable(f"{prefix}_{j}", lowBound=0, upBound=0)
        else:
            # Mixed: need binary variable
            y = pulp.LpVariable(f"{prefix}_{j}", lowBound=0, upBound=U)
            d = pulp.LpVariable(f"{prefix}_b_{j}", cat=pulp.LpBinary)
            prob += y >= z_vars[j]
            prob += y <= U * d
            prob += y <= z_vars[j] - L * (1 - d)
        y_vars.append(y)
    return y_vars


# ─────────────────────────────────────────────────────────────────────────────
# Network encoding
# ─────────────────────────────────────────────────────────────────────────────

def _milp_encode_sequential(prob, model, x_vars, x_lb, x_ub, prefix):
    """Encode nn.Sequential of Linear/ReLU layers as MILP constraints."""
    curr_vars = list(x_vars)
    curr_lb, curr_ub = x_lb.copy(), x_ub.copy()
    idx = 0
    for mod in model:
        if isinstance(mod, nn.Linear):
            W = mod.weight.detach().cpu().numpy().astype(np.float64)
            b = mod.bias.detach().cpu().numpy().astype(np.float64)
            out_lb, out_ub = _interval_bound_linear(W, b, curr_lb, curr_ub)
            curr_vars = _milp_add_linear(prob, curr_vars, W, b, f"{prefix}_L{idx}")
            curr_lb, curr_ub = out_lb, out_ub
            idx += 1
        elif isinstance(mod, nn.ReLU):
            curr_vars = _milp_add_relu(prob, curr_vars, curr_lb, curr_ub, f"{prefix}_R{idx}")
            curr_lb = np.maximum(curr_lb, 0)
            curr_ub = np.maximum(curr_ub, 0)
            idx += 1
        else:
            raise TypeError(
                f"unsupported layer for MILP encoding: {type(mod).__name__}"
            )
    return curr_vars, curr_lb, curr_ub


def _milp_encode_closed_loop(prob, f_pi_model, x_vars, x_lb, x_ub, prefix):
    """Encode ClosedLoopDynamicsModel as MILP constraints.

    Steps: normalize x -> pi(x) -> normalize u -> concat -> f_model -> denormalize
           -> Euler step -> ReLU-based state clipping.
    """
    x_mean = f_pi_model.x_mean.cpu().numpy().flatten().astype(np.float64)
    x_std  = f_pi_model.x_std.cpu().numpy().flatten().astype(np.float64)
    u_mean = f_pi_model.u_mean.cpu().numpy().flatten().astype(np.float64)
    u_std  = f_pi_model.u_std.cpu().numpy().flatten().astype(np.float64)
    dx_mean = f_pi_model.dx_mean.cpu().numpy().flatten().astype(np.float64)
    dx_std  = f_pi_model.dx_std.cpu().numpy().flatten().astype(np.float64)
    assert (x_std > 0).all() and (u_std > 0).all() and (dx_std > 0).all(), \
        "normalization std must be positive"
    xl = f_pi_model.x_low.cpu().numpy().flatten().astype(np.float64)
    xh = f_pi_model.x_high.cpu().numpy().flatten().astype(np.float64)
    x_dim = f_pi_model.x_dim
    u_dim = f_pi_model.u_dim

    # 1) x_norm = (x - x_mean) / x_std
    xn_vars, xn_lb, xn_ub = [], np.zeros(x_dim), np.zeros(x_dim)
    for i in range(x_dim):
        v = pulp.LpVariable(f"{prefix}_xn_{i}")
        prob += v == (x_vars[i] - x_mean[i]) * (1.0 / x_std[i])
        xn_vars.append(v)
        lo = (x_lb[i] - x_mean[i]) / x_std[i]
        hi = (x_ub[i] - x_mean[i]) / x_std[i]
        xn_lb[i], xn_ub[i] = min(lo, hi), max(lo, hi)

    # 2) u = pi(x)  (clipped LQR — nn.Sequential with ReLU)
    u_vars, u_lb, u_ub = _milp_encode_sequential(
        prob, f_pi_model.pi_model, x_vars, x_lb, x_ub, f"{prefix}_pi"
    )

    # 3) u_norm = (u - u_mean) / u_std
    un_vars, un_lb, un_ub = [], np.zeros(u_dim), np.zeros(u_dim)
    for i in range(u_dim):
        v = pulp.LpVariable(f"{prefix}_un_{i}")
        prob += v == (u_vars[i] - u_mean[i]) * (1.0 / u_std[i])
        un_vars.append(v)
        lo = (u_lb[i] - u_mean[i]) / u_std[i]
        hi = (u_ub[i] - u_mean[i]) / u_std[i]
        un_lb[i], un_ub[i] = min(lo, hi), max(lo, hi)

    # 4) xu_norm = concat(x_norm, u_norm)
    xu_vars = xn_vars + un_vars
    xu_lb = np.concatenate([xn_lb, un_lb])
    xu_ub = np.concatenate([xn_ub, un_ub])

    # 5) dx_norm = f_model(xu_norm)
    dxn_vars, dxn_lb, dxn_ub = _milp_encode_sequential(
        prob, f_pi_model.f_model, xu_vars, xu_lb, xu_ub, f"{prefix}_f"
    )

    # 6) dx = dx_norm * dx_std + dx_mean
    dx_vars, dx_lb, dx_ub = [], np.zeros(x_dim), np.zeros(x_dim)
    for i in range(x_dim):
        v = pulp.LpVariable(f"{prefix}_dx_{i}")
        prob += v == dxn_vars[i] * float(dx_std[i]) + float(dx_mean[i])
        dx_vars.append(v)
        lo = dxn_lb[i] * dx_std[i] + dx_mean[i]
        hi = dxn_ub[i] * dx_std[i] + dx_mean[i]
        dx_lb[i], dx_ub[i] = min(lo, hi), max(lo, hi)

    # 7) x_next_raw = x + dx
    xr_vars, xr_lb, xr_ub = [], np.zeros(x_dim), np.zeros(x_dim)
    for i in range(x_dim):
        v = pulp.LpVariable(f"{prefix}_xr_{i}")
        prob += v == x_vars[i] + dx_vars[i]
        xr_vars.append(v)
        xr_lb[i] = x_lb[i] + dx_lb[i]
        xr_ub[i] = x_ub[i] + dx_ub[i]

    # 8) clip lower: max(x_next, x_low) = relu(x_next - x_low) + x_low
    cl_in_vars, cl_lb, cl_ub = [], np.zeros(x_dim), np.zeros(x_dim)
    for i in range(x_dim):
        v = pulp.LpVariable(f"{prefix}_cl_{i}")
        prob += v == xr_vars[i] - float(xl[i])
        cl_in_vars.append(v)
        cl_lb[i] = xr_lb[i] - xl[i]
        cl_ub[i] = xr_ub[i] - xl[i]
    cl_relu = _milp_add_relu(prob, cl_in_vars, cl_lb, cl_ub, f"{prefix}_clR")
    xc1_vars, xc1_lb, xc1_ub = [], np.zeros(x_dim), np.zeros(x_dim)
    for i in range(x_dim):
        v = pulp.LpVariable(f"{prefix}_xc1_{i}")
        prob += v == cl_relu[i] + float(xl[i])
        xc1_vars.append(v)
        r_lb, r_ub = max(cl_lb[i], 0), max(cl_ub[i], 0)
        xc1_lb[i], xc1_ub[i] = r_lb + xl[i], r_ub + xl[i]

    # 9) clip upper: min(x_next, x_high) = -relu(-x_next + x_high) + x_high
    ch_in_vars, ch_lb, ch_ub = [], np.zeros(x_dim), np.zeros(x_dim)
    for i in range(x_dim):
        v = pulp.LpVariable(f"{prefix}_ch_{i}")
        prob += v == -1.0 * xc1_vars[i] + float(xh[i])
        ch_in_vars.append(v)
        ch_lb[i] = -xc1_ub[i] + xh[i]
        ch_ub[i] = -xc1_lb[i] + xh[i]
    ch_relu = _milp_add_relu(prob, ch_in_vars, ch_lb, ch_ub, f"{prefix}_chR")
    xn_vars2, xn_lb2, xn_ub2 = [], np.zeros(x_dim), np.zeros(x_dim)
    for i in range(x_dim):
        v = pulp.LpVariable(f"{prefix}_xnext_{i}")
        prob += v == -1.0 * ch_relu[i] + float(xh[i])
        xn_vars2.append(v)
        r_lb, r_ub = max(ch_lb[i], 0), max(ch_ub[i], 0)
        xn_lb2[i], xn_ub2[i] = -r_ub + xh[i], -r_lb + xh[i]

    return xn_vars2, xn_lb2, xn_ub2


def _make_solver(timeout, solver_kwargs):
    """Build the MILP solver. ``solver_kwargs['solver']`` (or the
    CERSYVE_MILP_SOLVER env var) picks any PuLP solver by name, e.g.
    "GUROBI" (default) or "PULP_CBC_CMD"; remaining kwargs pass through."""
    kw = dict(solver_kwargs or {})
    name = kw.pop("solver", os.environ.get("CERSYVE_MILP_SOLVER", "GUROBI"))
    if name == "GUROBI":
        # Gurobi grabs every core otherwise; CERSYVE_MILP_THREADS sets the per-solve cap
        kw.setdefault("threads", int(os.environ.get("CERSYVE_MILP_THREADS", "8")))
    return pulp.getSolver(name, msg=0, timeLimit=timeout, **kw)


def _extract_point(x_vars):
    """Read the solver's assignment. A variable absent from every active
    constraint has no assigned value; any point in its range will do."""
    values = [
        v.varValue if v.varValue is not None else (v.lowBound + v.upBound) / 2
        for v in x_vars
    ]
    return np.array([values], dtype=np.float32).T


def _confirm_witness(x_ce, V_model, other_model, x_low, x_high, *, invariance):
    """Re-evaluate a solver witness on the real networks, to catch an incumbent
    whose feasibility rests on the solver's own numerical slack rather than on
    the encoding.

    Tolerances are deliberately asymmetric. A spurious refutation only adds a
    harmless training point, while a spurious proof would invalidate the whole
    result — so this filter rejects only gross mismatches. ``V(x)`` is accepted
    up to ``TRAIN_MARGIN``, the same dilation the training loop already treats
    as inside the set: the MILP constrains ``V(x) <= 0`` in float64 and a
    boundary witness re-evaluated in float32 routinely lands a few 1e-6 above.
    Soundness of a reported proof comes from the exact encoding, never here.
    """
    x = np.clip(x_ce.astype(np.float32),
                np.asarray(x_low, dtype=np.float32)[:, None],
                np.asarray(x_high, dtype=np.float32)[:, None])
    # follow V_model's device: the caller may be training on the GPU, and h can
    # be an analytical model with no parameters to infer a device from
    device = next(V_model.parameters()).device
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        v = float(V_model(xt.T).T.cpu().numpy()[0, 0])
        if invariance:
            out = float(V_model(other_model(xt).T).T.cpu().numpy()[0, 0])
        else:
            out = float(other_model(xt.T).T.cpu().numpy()[0, 0])
    return x if (v <= TRAIN_MARGIN and out > VERIFY_MARGIN) else None


def _milp_result(prob, status, x_vars):
    """Interpret a solved max-violation MILP.

    Returns (found_violation, counterexample). ``False`` (no violation exists)
    requires a proof — infeasibility, or a proven optimum at or below the
    margin. ``True`` only needs a witness: any feasible point with objective
    above the margin violates the property by construction, so an incumbent
    from a timed-out solve is just as sound as a proven optimum. Reporting
    such an incumbent as inconclusive threw away counterexamples the solver
    already held (robot_arm: 59 calls, 4.9 h of solver time, no information),
    and an inconclusive verdict teaches the training loop nothing.
    """
    if status == pulp.constants.LpStatusInfeasible:
        return False, None

    obj = pulp.value(prob.objective)
    proved = (status == pulp.constants.LpStatusOptimal
              and prob.sol_status == pulp.constants.LpSolutionOptimal)

    if obj is not None and obj > VERIFY_MARGIN:
        return True, _extract_point(x_vars)   # witness — sound either way
    if proved and obj is not None:
        return False, None                    # proven optimum within margin
    return None, None                         # no proof, no witness


# ─────────────────────────────────────────────────────────────────────────────
# Public verification API
# ─────────────────────────────────────────────────────────────────────────────

def _note(record, stage, found, status, prob, seconds):
    if record is not None:
        try:
            obj = pulp.value(prob.objective)
        except Exception:  # pragma: no cover - solver without a value
            obj = None
        record.append({"stage": stage,
                       "verdict": {True: "violation", False: "pass", None: "inconclusive"}[found],
                       "status": pulp.LpStatus.get(status, str(status)),
                       "objective": None if obj is None else float(obj),
                       "seconds": float(seconds)})


def milp_verify_constraint(V_model, h_model, x_low, x_high, timeout=300,
                           solver_kwargs=None, record=None):
    """Check via MILP if there exists x with V(x) <= 0 AND h(x) > 0.

    Args:
        V_model: Value function (nn.Sequential, Linear+ReLU)
        h_model: Constraint model (nn.Sequential, Linear+ReLU)
        x_low, x_high: State space bounds (np.ndarray)
        timeout: Solver time limit in seconds

    Returns:
        found_violation: True (violation found), False (verified safe), None (inconclusive)
        counterexample: np.ndarray [x_dim, 1] or None
    """
    xl = np.asarray(x_low, dtype=np.float64)
    xh = np.asarray(x_high, dtype=np.float64)
    x_dim = len(xl)

    prob = pulp.LpProblem("con_verify", pulp.LpMaximize)
    x_vars = [
        pulp.LpVariable(f"x_{i}", lowBound=float(xl[i]), upBound=float(xh[i]))
        for i in range(x_dim)
    ]

    # Encode V(x) and constrain V(x) <= 0
    v_vars, _, _ = _milp_encode_sequential(prob, V_model, x_vars, xl, xh, "V")
    prob += v_vars[0] <= 0

    # Encode h(x)
    h_vars, _, _ = _milp_encode_sequential(prob, h_model, x_vars, xl, xh, "h")

    # Maximize h(x) subject to V(x) <= 0
    prob += h_vars[0]
    solver = _make_solver(timeout, solver_kwargs)
    t0 = time.perf_counter()
    status = prob.solve(solver)
    found, x_ce = _milp_result(prob, status, x_vars)
    if found is True:
        x_ce = _confirm_witness(x_ce, V_model, h_model, xl, xh, invariance=False)
        if x_ce is None:
            found = None
    _note(record, "constraint", found, status, prob, time.perf_counter() - t0)
    return found, x_ce


def milp_verify_invariance(V_model, f_pi_model, x_low, x_high, timeout=300,
                           solver_kwargs=None, record=None):
    """Check via MILP if there exists x with V(x) <= 0 AND V(f_pi(x)) > 0.

    Args:
        V_model: Value function (nn.Sequential, Linear+ReLU)
        f_pi_model: Closed-loop dynamics (ClosedLoopDynamicsModel)
        x_low, x_high: State space bounds (np.ndarray)
        timeout: Solver time limit in seconds

    Returns:
        found_violation: True (violation found), False (verified safe), None (inconclusive)
        counterexample: np.ndarray [x_dim, 1] or None
    """
    xl = np.asarray(x_low, dtype=np.float64)
    xh = np.asarray(x_high, dtype=np.float64)
    x_dim = len(xl)

    prob = pulp.LpProblem("inv_verify", pulp.LpMaximize)
    x_vars = [
        pulp.LpVariable(f"x_{i}", lowBound=float(xl[i]), upBound=float(xh[i]))
        for i in range(x_dim)
    ]

    # Encode V(x) and constrain V(x) <= 0
    v_vars, _, _ = _milp_encode_sequential(prob, V_model, x_vars, xl, xh, "V")
    prob += v_vars[0] <= 0

    # Encode f_pi(x) -> x_next
    xn_vars, xn_lb, xn_ub = _milp_encode_closed_loop(
        prob, f_pi_model, x_vars, xl, xh, "fpi"
    )

    # Encode V(x_next)
    vn_vars, _, _ = _milp_encode_sequential(prob, V_model, xn_vars, xn_lb, xn_ub, "Vn")

    # Maximize V(x_next) subject to V(x) <= 0
    prob += vn_vars[0]
    solver = _make_solver(timeout, solver_kwargs)
    t0 = time.perf_counter()
    status = prob.solve(solver)
    found, x_ce = _milp_result(prob, status, x_vars)
    if found is True:
        x_ce = _confirm_witness(x_ce, V_model, f_pi_model, xl, xh, invariance=True)
        if x_ce is None:
            found = None
    _note(record, "invariance", found, status, prob, time.perf_counter() - t0)
    return found, x_ce


def milp_verify(V_model, h_model, f_pi_model, x_low, x_high, timeout=300,
                verbose=True, solver_kwargs=None, record=None):
    """Run both constraint and invariance MILP checks.

    Args:
        V_model: Value function model
        h_model: Constraint model
        f_pi_model: Closed-loop dynamics model
        x_low, x_high: State space bounds
        timeout: Solver time limit per sub-problem (seconds)
        verbose: Print progress

    Returns:
        verified: True (formally verified), False (violation found), None (inconclusive)
        counterexamples: np.ndarray [x_dim, n_ce] or None
    """
    V_model.eval()
    all_ce = []
    verified = True

    # Constraint check
    t0 = time.time()
    found, ce = milp_verify_constraint(V_model, h_model, x_low, x_high, timeout=timeout,
                                       solver_kwargs=solver_kwargs, record=record)
    dt = time.time() - t0
    if found is True:
        if verbose:
            print(f"    MILP constraint check: VIOLATION found  ({dt:.1f}s)")
        all_ce.append(ce)
        verified = False
    elif found is False:
        if verbose:
            print(f"    MILP constraint check: PASSED           ({dt:.1f}s)")
    else:
        if verbose:
            print(f"    MILP constraint check: INCONCLUSIVE     ({dt:.1f}s)")
        verified = None

    # Invariance check
    t0 = time.time()
    found, ce = milp_verify_invariance(V_model, f_pi_model, x_low, x_high, timeout=timeout,
                                       solver_kwargs=solver_kwargs, record=record)
    dt = time.time() - t0
    if found is True:
        if verbose:
            print(f"    MILP invariance check: VIOLATION found  ({dt:.1f}s)")
        all_ce.append(ce)
        verified = False
    elif found is False:
        if verbose:
            print(f"    MILP invariance check: PASSED           ({dt:.1f}s)")
    else:
        if verbose:
            print(f"    MILP invariance check: INCONCLUSIVE     ({dt:.1f}s)")
        if verified is True:
            verified = None

    if all_ce:
        return verified, np.concatenate(all_ce, axis=1)
    return verified, None


# ============================================================================
# Part 2: direct Gurobi verifier with LP-tightened bounds and box splitting
# ============================================================================

"""Faster exact MILP verification (opt-in; cersyve/milp.py stays the reference).

Same questions and the same verdict semantics as cersyve.milp:

    constraint:  exists x in box, V(x) <= 0, h(x)       > VERIFY_MARGIN ?
    invariance:  exists x in box, V(x) <= 0, V(f_pi(x)) > VERIFY_MARGIN ?

Three accelerations, all sound:

1. Tighter big-M bounds. Every pre-activation that the interval bound leaves
   unstable is re-bounded by the LP relaxation of everything before it (triangle
   relaxation of earlier ReLUs, plus V(x) <= 0 once that copy of V is built),
   padded outward by a numerical safety margin and intersected with the interval
   bound. A ReLU whose tightened bound no longer straddles zero loses its binary.
2. Input-box splitting. The box is cut into K sub-boxes by longest-edge bisection;
   each gets its own bounds and MILP, solved in parallel with a few threads each.
   PASS needs every sub-box to pass (the sub-boxes cover the box); a confirmed
   witness from any sub-box refutes, and the remaining solves are told to stop.
3. Bound-based early stop. A solve that proves ObjBound <= VERIFY_MARGIN stops at
   once (a proof); optionally a solve stops at the first incumbent above
   `best_obj_stop` (a refutation, as milp__BestObjStop does for cersyve.milp).

Verdicts per solve (identical to cersyve.milp._milp_result):
    infeasible                                   -> no violation (proof)
    incumbent objective > VERIFY_MARGIN          -> violation (witness, confirmed
                                                    by cersyve.milp._confirm_witness)
    proven bound <= VERIFY_MARGIN, or OPTIMAL
      with objective <= VERIFY_MARGIN            -> no violation (proof)
    anything else (time limit, numerics)         -> inconclusive

Selecting it (not wired into any default):
    verifier=milp_fast  with  milp_fast__split=32  milp_fast__bounds=lp
    milp_fast__threads=2  milp_fast__workers=32  milp_fast__best_obj_stop=1e-4
"""

import multiprocessing as mp
import os
import time

import numpy as np
import torch.nn as nn

from .tolerances import VERIFY_MARGIN


# ─────────────────────────────────────────────────────────────────────────────
# Graph: a flat list of scalar nodes, each an input, an affine combination of
# earlier nodes, or a ReLU of one earlier node. Plain numpy so it pickles cheaply.
# ─────────────────────────────────────────────────────────────────────────────

class Graph:
    def __init__(self, x_dim):
        self.kind = []        # 'in' | 'lin' | 'relu'
        self.arg = []         # in: index | lin: (src ids, coefs, const) | relu: src id
        self.inputs = []
        for i in range(x_dim):
            self.inputs.append(self._add('in', i))
        self.side = []        # (node id, upper bound): node <= ub constraints
        self.objective = None

    def _add(self, kind, arg):
        self.kind.append(kind)
        self.arg.append(arg)
        return len(self.kind) - 1

    def affine(self, src, W, b):
        W = np.asarray(W, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        src = np.asarray(src, dtype=np.int64)
        out = []
        for j in range(W.shape[0]):
            nz = np.nonzero(W[j])[0]
            out.append(self._add('lin', (src[nz], W[j, nz].copy(), float(b[j]))))
        return out

    def relu(self, src):
        return [self._add('relu', int(s)) for s in src]

    def sequential(self, model, src):
        cur = list(src)
        for mod in model:
            if isinstance(mod, nn.Linear):
                cur = self.affine(cur, mod.weight.detach().cpu().numpy(),
                                  mod.bias.detach().cpu().numpy())
            elif isinstance(mod, nn.ReLU):
                cur = self.relu(cur)
            else:
                raise TypeError(f"unsupported layer: {type(mod).__name__}")
        return cur

    def closed_loop(self, f_pi_model, x):
        """Exactly the operations of cersyve.milp._milp_encode_closed_loop."""
        g = lambda t: t.cpu().numpy().flatten().astype(np.float64)
        x_mean, x_std = g(f_pi_model.x_mean), g(f_pi_model.x_std)
        u_mean, u_std = g(f_pi_model.u_mean), g(f_pi_model.u_std)
        dx_mean, dx_std = g(f_pi_model.dx_mean), g(f_pi_model.dx_std)
        xl, xh = g(f_pi_model.x_low), g(f_pi_model.x_high)
        n, m = f_pi_model.x_dim, f_pi_model.u_dim
        I = np.eye(n)
        xn = self.affine(x, I / x_std[:, None], -x_mean / x_std)
        u = self.sequential(f_pi_model.pi_model, x)
        un = self.affine(u, np.eye(m) / u_std[:, None], -u_mean / u_std)
        dxn = self.sequential(f_pi_model.f_model, xn + un)
        dx = self.affine(dxn, np.diag(dx_std), dx_mean)
        xr = [self._add('lin', (np.array([x[i], dx[i]]), np.array([1.0, 1.0]), 0.0))
              for i in range(n)]
        cl = self.relu(self.affine(xr, I, -xl))
        xc1 = self.affine(cl, I, xl)
        ch = self.relu(self.affine(xc1, -I, xh))
        return self.affine(ch, -I, xh)


def build_graph(V_model, other_model, x_dim, invariance):
    """Graph of the verification problem: V(x) <= 0 as a side constraint and the
    violation quantity (V(f(x)) or h(x)) as the objective to maximise."""
    G = Graph(x_dim)
    v = G.sequential(V_model, G.inputs)
    G.side.append((v[0], 0.0))
    if invariance:
        xn = G.closed_loop(other_model, G.inputs)
        out = G.sequential(V_model, xn)
    else:
        out = G.sequential(other_model, G.inputs)
    G.objective = out[0]
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Bounds
# ─────────────────────────────────────────────────────────────────────────────

def _lin_interval(arg, lb, ub):
    src, c, const = arg
    lo = np.where(c > 0, c * lb[src], c * ub[src]).sum() + const
    hi = np.where(c > 0, c * ub[src], c * lb[src]).sum() + const
    return lo, hi


def interval_bounds(G, box_lb, box_ub):
    N = len(G.kind)
    lb, ub = np.zeros(N), np.zeros(N)
    for k in range(N):
        kind, arg = G.kind[k], G.arg[k]
        if kind == 'in':
            lb[k], ub[k] = box_lb[arg], box_ub[arg]
        elif kind == 'lin':
            lb[k], ub[k] = _lin_interval(arg, lb, ub)
        else:
            lb[k], ub[k] = max(lb[arg], 0.0), max(ub[arg], 0.0)
    return lb, ub


def _pad(v):
    # outward safety margin for LP-derived bounds (simplex tolerances are ~1e-9
    # at the settings used; this is several orders above that)
    return 1e-6 * (1.0 + abs(v))


def lp_bounds(G, box_lb, box_ub, env, time_budget=None):
    """Layer-ordered LP tightening. Returns (lb, ub, n_lp, feasible).

    Each node is added to one incremental LP as it is reached; before a ReLU is
    relaxed, its source is maximised and minimised over the relaxation so far
    (only if the interval bound is unstable). Side constraints (V(x) <= 0) are
    added as soon as their node exists, so later copies are bounded under them.
    feasible=False means the relaxation itself is infeasible, which makes the
    MILP infeasible too (a proof); callers still run the MILP for the verdict.
    """
    import gurobipy as gp
    from gurobipy import GRB
    N = len(G.kind)
    lb, ub = np.zeros(N), np.zeros(N)
    m = gp.Model(env=env)
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.Method = 1                   # dual simplex, warm-started across objectives
    m.Params.FeasibilityTol = 1e-9
    m.Params.OptimalityTol = 1e-9
    side = {nid: val for nid, val in G.side}
    var = [None] * N
    n_lp = 0
    t0 = time.time()
    for k in range(N):
        kind, arg = G.kind[k], G.arg[k]
        if kind == 'in':
            lb[k], ub[k] = box_lb[arg], box_ub[arg]
            var[k] = m.addVar(lb=lb[k], ub=ub[k])
        elif kind == 'lin':
            lb[k], ub[k] = _lin_interval(arg, lb, ub)
            src, c, const = arg
            var[k] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m.addLConstr(gp.LinExpr(c.tolist(), [var[s] for s in src]) + const,
                         GRB.EQUAL, var[k])
        else:
            s = arg
            if lb[s] < 0.0 < ub[s] and (time_budget is None or time.time() - t0 < time_budget):
                for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                    m.setObjective(var[s], sense)
                    m.optimize()
                    n_lp += 1
                    if m.Status == GRB.INFEASIBLE:
                        return lb, ub, n_lp, False
                    if m.Status != GRB.OPTIMAL:
                        continue          # keep the interval bound
                    val = m.ObjVal
                    if sense == GRB.MAXIMIZE:
                        ub[s] = min(ub[s], val + _pad(val))
                    else:
                        lb[s] = max(lb[s], val - _pad(val))
                    if lb[s] > ub[s]:     # numerically empty: fall back safely
                        lo, hi = _lin_interval(G.arg[s], lb, ub) if G.kind[s] == 'lin' \
                            else (lb[s], ub[s])
                        lb[s], ub[s] = min(lo, hi), max(lo, hi)
                var[s].LB, var[s].UB = lb[s], ub[s]
            lb[k], ub[k] = max(lb[s], 0.0), max(ub[s], 0.0)
            y = m.addVar(lb=lb[k], ub=ub[k])
            if lb[s] >= 0.0:
                m.addLConstr(y, GRB.EQUAL, var[s])
            elif ub[s] <= 0.0:
                pass                       # y fixed to [0, 0] by its bounds
            else:
                m.addLConstr(y, GRB.GREATER_EQUAL, var[s])
                slope = ub[s] / (ub[s] - lb[s])
                m.addLConstr(y, GRB.LESS_EQUAL, slope * var[s] - slope * lb[s])
            var[k] = y
        if k in side:
            m.addLConstr(var[k], GRB.LESS_EQUAL, side[k])
            ub[k] = min(ub[k], side[k])
    m.dispose()
    return lb, ub, n_lp, True


# ─────────────────────────────────────────────────────────────────────────────
# MILP on one box
# ─────────────────────────────────────────────────────────────────────────────

_STOP = None


def _init_worker(stop_event):
    global _STOP
    _STOP = stop_event


def solve_box(G, box_lb, box_ub, opts):
    """Exact MILP over one box. Returns a dict with verdict in
    {'pass', 'violation', 'inconclusive'}, the witness x, timings and counts."""
    import gurobipy as gp
    from gurobipy import GRB
    t0 = time.time()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        ilb, iub = interval_bounds(G, box_lb, box_ub)
        n_relu = sum(1 for k in G.kind if k == 'relu')
        unstable_int = sum(1 for k, a in zip(G.kind, G.arg)
                           if k == 'relu' and ilb[a] < 0.0 < iub[a])
        n_lp = 0
        if opts.get("bounds", "interval") == "lp":
            lb, ub, n_lp, feasible = lp_bounds(G, box_lb, box_ub, env,
                                               time_budget=opts.get("lp_time_budget"))
            if not feasible:
                lb, ub = ilb, iub          # MILP below decides (it is infeasible too)
        else:
            lb, ub = ilb, iub
        t_bounds = time.time() - t0
        if opts.get("precheck", True) and ub[G.objective] <= VERIFY_MARGIN:
            # bound pre-check: the (LP or interval) relaxation already bounds the
            # violation below the margin on this cell, so the exact MILP is skipped.
            # Sound: every relaxation bound is an upper bound of the true maximum.
            return dict(verdict="pass", x=None, obj=None, bound=float(ub[G.objective]), status=-1,
                        t_bounds=t_bounds, t_total=time.time() - t0, relu=n_relu,
                        unstable_interval=unstable_int, unstable=0, n_lp=n_lp, t_pool=0.0,
                        prechecked=True)

        m = gp.Model(env=env)
        m.Params.OutputFlag = 0
        m.Params.Threads = int(opts.get("threads", 2))
        m.Params.TimeLimit = float(opts.get("timeout", 300))
        m.Params.BestBdStop = VERIFY_MARGIN - 1e-6   # proof: bound strictly below the margin
        if opts.get("best_obj_stop") is not None:
            m.Params.BestObjStop = float(opts["best_obj_stop"])
        for key, val in (opts.get("gurobi_params") or {}).items():
            m.setParam(key, val)
        N = len(G.kind)
        var = [None] * N
        unstable = 0
        side = {nid: val for nid, val in G.side}
        for k in range(N):
            kind, arg = G.kind[k], G.arg[k]
            if kind == 'in':
                var[k] = m.addVar(lb=box_lb[arg], ub=box_ub[arg])
            elif kind == 'lin':
                src, c, const = arg
                var[k] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                m.addLConstr(gp.LinExpr(c.tolist(), [var[s] for s in src]) + const,
                             GRB.EQUAL, var[k])
            else:
                s = arg
                L, U = lb[s], ub[s]
                if L >= 0.0:
                    y = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    m.addLConstr(y, GRB.EQUAL, var[s])
                elif U <= 0.0:
                    y = m.addVar(lb=0.0, ub=0.0)
                else:
                    unstable += 1
                    y = m.addVar(lb=0.0, ub=U)
                    d = m.addVar(vtype=GRB.BINARY)
                    m.addLConstr(y, GRB.GREATER_EQUAL, var[s])
                    m.addLConstr(y, GRB.LESS_EQUAL, U * d)
                    m.addLConstr(y, GRB.LESS_EQUAL, var[s] - L * (1 - d))
                var[k] = y
            if k in side:
                m.addLConstr(var[k], GRB.LESS_EQUAL, side[k])
        xs = opts.get("mip_start")
        if xs is not None:
            xs = np.asarray(xs, dtype=np.float64).reshape(-1)
            if xs.shape[0] == len(G.inputs) and np.all(xs >= np.asarray(box_lb) - 1e-9) \
                    and np.all(xs <= np.asarray(box_ub) + 1e-9):
                # input-only MIP start: Gurobi completes the rest (a start with a
                # high objective lets BestObjStop fire on the first incumbent)
                for j, nid in enumerate(G.inputs):
                    var[nid].Start = float(min(max(xs[j], box_lb[j]), box_ub[j]))
        m.setObjective(var[G.objective], GRB.MAXIMIZE)

        def cb(model, where):
            if _STOP is not None and _STOP.is_set():
                model.terminate()
        m.optimize(cb if _STOP is not None else None)
        if (m.Status == GRB.USER_OBJ_LIMIT and not (m.SolCount > 0 and m.ObjVal > VERIFY_MARGIN)
                and not m.ObjBound <= VERIFY_MARGIN):
            # BestBdStop fires within Gurobi's tolerance (bound 1e-17 above the margin
            # was seen); that is not a proof, so finish the solve to optimality and
            # fall back to the reference semantics
            m.Params.BestBdStop = -GRB.INFINITY
            m.optimize(cb if _STOP is not None else None)
        if m.Status == GRB.INF_OR_UNBD:
            # every variable is bounded through the box, so this is infeasibility
            # hidden by dual reductions; re-solve to get an explicit status
            m.Params.DualReductions = 0
            m.optimize(cb if _STOP is not None else None)

        st = m.Status
        x = None
        obj = m.ObjVal if m.SolCount > 0 else None
        bound = m.ObjBound if st not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else None
        t_pool = 0.0
        if st == GRB.INFEASIBLE:
            verdict = "pass"
        elif obj is not None and obj > VERIFY_MARGIN:
            verdict = "violation"
            x = np.array([[var[i].X for i in G.inputs]], dtype=np.float64).T
            pool = opts.get("pool")
            if pool:
                # Refutation-only harvest: a SEPARATE re-solve of this box, run only
                # after it has already refuted, so no proving solve is ever slowed.
                # Cutoff drops every solution at or below the margin; each pooled x
                # is still re-confirmed on the real networks by the caller.
                tp = time.time()
                m.Params.BestObjStop = GRB.INFINITY
                m.Params.BestBdStop = -GRB.INFINITY
                m.Params.PoolSearchMode = 2
                m.Params.PoolSolutions = int(pool.get("n", 32))
                m.Params.TimeLimit = float(pool.get("time", 5.0))
                m.Params.Cutoff = VERIFY_MARGIN
                m.reset(0)
                m.optimize()
                xs = [x]
                for sn in range(m.SolCount):
                    m.Params.SolutionNumber = sn
                    if m.PoolObjVal > VERIFY_MARGIN:
                        xs.append(np.array([[var[i].Xn for i in G.inputs]], dtype=np.float64).T)
                x = np.concatenate(xs, axis=1)
                t_pool = time.time() - tp
        elif bound is not None and bound <= VERIFY_MARGIN and st in (
                GRB.OPTIMAL, GRB.USER_OBJ_LIMIT):
            verdict = "pass"
        elif st == GRB.OPTIMAL and obj is not None and obj <= VERIFY_MARGIN:
            verdict = "pass"               # cersyve.milp semantics (proven optimum)
        else:
            verdict = "inconclusive"
        out = dict(verdict=verdict, x=x, obj=obj, bound=bound, status=st,
                   t_bounds=t_bounds, t_total=time.time() - t0,
                   relu=n_relu, unstable_interval=unstable_int, unstable=unstable,
                   n_lp=n_lp, t_pool=t_pool)
        if verdict == "violation" and _STOP is not None and opts.get("stop_on_violation", True):
            _STOP.set()
        m.dispose()
        return out
    finally:
        env.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# Box splitting and the public API
# ─────────────────────────────────────────────────────────────────────────────

def split_box(lb, ub, K):
    """Longest-edge (relative to the full box) bisection into K boxes (K a power of 2)."""
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    width0 = np.maximum(ub - lb, 1e-12)
    boxes = [(lb.copy(), ub.copy())]
    while len(boxes) < K:
        nxt = []
        for bl, bu in boxes:
            d = int(np.argmax((bu - bl) / width0))
            mid = 0.5 * (bl[d] + bu[d])
            l2, u1 = bl.copy(), bu.copy()
            u1[d] = mid
            l2[d] = mid
            nxt += [(bl, u1), (l2, bu)]
        boxes = nxt
    return boxes


def refine_split(boxes, regions, depth, max_boxes):
    """Enumeration-guided refinement: every cell of the split that meets one of the
    enumerated regions ([K, d, 2] lows/highs, the leaves the enumeration flagged) is
    bisected `depth` more times along its longest edge. Cells away from the flagged
    leaves stay coarse and are usually settled by the bound pre-check without a MILP.
    The result is still a cover of the box, so the verdict semantics are unchanged."""
    reg = np.asarray(regions, dtype=np.float64)
    if reg.ndim != 3 or reg.shape[0] == 0:
        return boxes
    rl, rh = reg[:, :, 0], reg[:, :, 1]
    width0 = np.maximum(boxes[0][1] - boxes[0][0], 1e-12)
    for lo, hi in boxes[1:]:
        width0 = np.maximum(width0, hi - lo)
    out = list(boxes)
    for _ in range(depth):
        nxt = []
        for bl, bu in out:
            hit = bool(np.any(np.all((bl <= rh) & (bu >= rl), axis=1)))
            if hit and len(out) + len(nxt) < max_boxes:
                d = int(np.argmax((bu - bl) / width0))
                mid = 0.5 * (bl[d] + bu[d])
                l2, u1 = bl.copy(), bu.copy()
                u1[d] = mid
                l2[d] = mid
                nxt += [(bl, u1), (l2, bu)]
            else:
                nxt.append((bl, bu))
        out = nxt
    return out


_POOL = {}


def _pool(workers):
    if workers not in _POOL:
        ctx = mp.get_context("spawn")
        stop = ctx.Event()
        _POOL[workers] = (ctx.Pool(workers, initializer=_init_worker, initargs=(stop,)), stop)
    return _POOL[workers]


def check(V_model, other_model, x_low, x_high, invariance, timeout=300, split=1,
          bounds="lp", threads=2, workers=None, best_obj_stop=None, gurobi_params=None,
          lp_time_budget=None, collect_all=False, harvest_time=None, pool=None,
          mip_start=None, refine_boxes=None, refine_depth=2, max_boxes=512, precheck=True):
    """One check (constraint or invariance) with the cersyve.milp return convention:
    (True, witness) / (False, None) / (None, None), plus a stats dict."""
    from .milp import _confirm_witness
    xl = np.asarray(x_low, dtype=np.float64)
    xh = np.asarray(x_high, dtype=np.float64)
    G = build_graph(V_model, other_model, len(xl), invariance)
    opts = dict(bounds=bounds, threads=threads, timeout=timeout, best_obj_stop=best_obj_stop,
                gurobi_params=gurobi_params, lp_time_budget=lp_time_budget,
                stop_on_violation=not collect_all, pool=pool, mip_start=mip_start,
                precheck=precheck)
    boxes = split_box(xl, xh, split)
    if refine_boxes is not None and refine_depth > 0:
        boxes = refine_split(boxes, refine_boxes, refine_depth, max_boxes)
    t0 = time.time()
    if split == 1:
        global _STOP
        _STOP = None
        results = [solve_box(G, boxes[0][0], boxes[0][1], opts)]
    else:
        pool, stop = _pool(workers or split)
        stop.clear()
        pending = [pool.apply_async(solve_box, (G, bl, bu, opts)) for bl, bu in boxes]
        results = [None] * len(pending)
        deadline = None
        while any(r is None for r in results):
            for i, p in enumerate(pending):
                if results[i] is None and p.ready():
                    results[i] = p.get()
                    if collect_all and deadline is None and results[i]["verdict"] == "violation":
                        # harvest: other boxes get harvest_time more seconds, then stop
                        deadline = time.time() + float(harvest_time or 0.0)
            if deadline is not None and time.time() > deadline:
                stop.set()
            time.sleep(0.01)
        stop.clear()
    wall = time.time() - t0
    witnesses, seen = [], set()
    for r in results:
        if r["verdict"] == "violation":
            for j in range(r["x"].shape[1]):
                w = _confirm_witness(r["x"][:, j:j + 1].astype(np.float32), V_model,
                                     other_model, xl, xh, invariance=invariance)
                if w is None:
                    continue
                key = tuple(np.round(w[:, 0], 6))
                if key not in seen:
                    seen.add(key)
                    witnesses.append(w)
    objs = [r["obj"] for r in results if r["verdict"] == "violation" and r["obj"] is not None]
    stats = dict(wall=wall, boxes=len(boxes), max_obj=(max(objs) if objs else None),
                 prechecked=int(sum(1 for r in results if r.get("prechecked"))),
                 verdicts={v: sum(r["verdict"] == v for r in results)
                           for v in ("pass", "violation", "inconclusive")},
                 unstable_interval=int(sum(r["unstable_interval"] for r in results)),
                 unstable=int(sum(r["unstable"] for r in results)),
                 relu=int(results[0]["relu"]) * len(boxes),
                 t_bounds_sum=float(sum(r["t_bounds"] for r in results)),
                 max_box_s=float(max(r["t_total"] for r in results)),
                 n_lp=int(sum(r["n_lp"] for r in results)),
                 t_pool_sum=float(sum(r["t_pool"] for r in results)),
                 raw_witnesses=int(sum(r["x"].shape[1] for r in results if r["x"] is not None)),
                 confirmed_witnesses=len(witnesses))
    if witnesses:
        many = collect_all or bool(pool)
        return True, (np.concatenate(witnesses, axis=1) if many else witnesses[0]), stats
    # A terminated sibling solve is inconclusive only because a violation was found
    # elsewhere; with no confirmed witness, any inconclusive box means no proof.
    if all(r["verdict"] == "pass" for r in results):
        return False, None, stats
    return None, None, stats


def milp_fast_verify(V_model, h_model, f_pi_model, x_low, x_high, timeout=300,
                     verbose=True, mip_start=None, refine_boxes=None, record=None, **kw):
    """Drop-in for cersyve.milp.milp_verify: (verified, counterexamples).
    ``mip_start``: optional {"constraint": x, "invariance": x} input states handed
    to Gurobi as MIP starts (a sound hint; verdict semantics unchanged)."""
    V_model.eval()
    all_ce, verified = [], True
    for name, other, inv in (("constraint", h_model, False), ("invariance", f_pi_model, True)):
        t0 = time.time()
        found, ce, st = check(V_model, other, x_low, x_high, inv, timeout=timeout,
                              mip_start=(mip_start or {}).get(name), refine_boxes=refine_boxes, **kw)
        dt = time.time() - t0
        if found is True:
            all_ce.append(ce)
            verified = False
            msg = "VIOLATION found" + (f" obj={st['max_obj']:.2e}" if st.get("max_obj") is not None else "")
        elif found is False:
            msg = "PASSED"
        else:
            msg = "INCONCLUSIVE"
            if verified is True:
                verified = None
        if record is not None:
            record.append({"stage": name, "verdict": {True: "violation", False: "pass", None: "inconclusive"}[found],
                           "status": f"boxes={st.get('boxes')} prechecked={st.get('prechecked')} "
                                     f"verdicts={st.get('verdicts')}",
                           "objective": st.get("max_obj"), "seconds": float(dt), "boxes": st.get("boxes"),
                           "n_lp": st.get("n_lp")})
        if verbose:
            print(f"    MILP-fast {name} check: {msg:16s} ({dt:.1f}s, {st.get('prechecked', 0)} of "
                  f"{st.get('boxes', 0)} cells by bounds)")
    if all_ce:
        return verified, np.concatenate(all_ce, axis=1)
    return verified, None
