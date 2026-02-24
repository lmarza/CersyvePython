"""MILP formal verification for ReLU neural network CBFs.

Encodes piecewise-linear (ReLU) neural networks as Mixed-Integer Linear Programs
and solves for counterexamples to constraint satisfaction and forward invariance.

Uses PuLP + CBC (open-source, no license required).
"""

import time
import numpy as np
import torch.nn as nn
import pulp


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


# ─────────────────────────────────────────────────────────────────────────────
# Public verification API
# ─────────────────────────────────────────────────────────────────────────────

def milp_verify_constraint(V_model, h_model, x_low, x_high, timeout=300):
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
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout)
    status = prob.solve(solver)

    if status == pulp.constants.LpStatusOptimal:
        obj = pulp.value(prob.objective)
        if obj is not None and obj > 1e-6:
            x_ce = np.array(
                [[x_vars[i].varValue for i in range(x_dim)]], dtype=np.float32
            ).T
            return True, x_ce
        return False, None
    if status == pulp.constants.LpStatusInfeasible:
        return False, None
    return None, None


def milp_verify_invariance(V_model, f_pi_model, x_low, x_high, timeout=300):
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
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout)
    status = prob.solve(solver)

    if status == pulp.constants.LpStatusOptimal:
        obj = pulp.value(prob.objective)
        if obj is not None and obj > 1e-6:
            x_ce = np.array(
                [[x_vars[i].varValue for i in range(x_dim)]], dtype=np.float32
            ).T
            return True, x_ce
        return False, None
    if status == pulp.constants.LpStatusInfeasible:
        return False, None
    return None, None


def milp_verify(V_model, h_model, f_pi_model, x_low, x_high, timeout=300,
                verbose=True):
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
    found, ce = milp_verify_constraint(V_model, h_model, x_low, x_high, timeout=timeout)
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
    found, ce = milp_verify_invariance(V_model, f_pi_model, x_low, x_high, timeout=timeout)
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
