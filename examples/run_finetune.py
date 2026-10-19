"""Fine-tuning runner for the five benchmarks.

Runs one task/method pair over one or more seeds with the hyperparameters in
configs/ and writes per-seed and aggregate JSON results under the task's model
directory:

    models/<task>/results/<method>[_<tag>]/seed<k>.json   result records
    models/<task>/checkpoints/V_finetune_<method>[_<tag>]_seed<k>.pth
    models/<task>/logs/<run>/                               TensorBoard + periodic checkpoints

    python examples/run_finetune.py --task pendulum --method enum_milp --seeds 1 0 2
    python examples/run_finetune.py --task robot_arm --method pgd_milp --override max_skip=300
    python examples/run_finetune.py --task cartpole --method enum_milp --config configs/final/cartpole_enum_milp.json

Accounting (identical for both methods, measured at the verifier boundary inside
the training loop plus the final reference re-verification done here):

    verifier_calls   invocations of the verifier function (training + 1 reference)
    milp_solves      MILP solver runs started by those invocations (<= 2 per call)
    milp_seconds     wall-clock seconds elapsed inside the invocations, summed
    wall_time_s      wall-clock seconds of the training loop
    certified_wall_s seconds from the start of training to the reference proof
    end_to_end_wall_s seconds for the whole seed (model loading, training,
                     re-verification, TFR evaluation)
"""

import argparse
import ast
import functools
import importlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

import cersyve
from cersyve import (
    create_closed_loop_dynamics_model,
    estimate_tfr,
    finetune_value_w_MILP,
    finetune_value_enum_MILP,
    load_mlp,
)
from cersyve.milp import milp_verify
from configs import METHODS, RUNNER_KEYS, TASKS, get_config

TRAINERS = {
    "pgd_milp": finetune_value_w_MILP,
    "enum_milp": finetune_value_enum_MILP,
    "enum_lean": finetune_value_enum_MILP,   # alias of the canonical implementation
}


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        try:
            overrides[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[key] = value
    return overrides


def resolve_system(spec: str):
    module_name, attr = spec.split(":")
    return getattr(importlib.import_module(module_name), attr)


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def task_dirs(task_info):
    base = os.path.join(REPO_ROOT, task_info["model_dir"])
    dirs = {"base": base,
            "results": os.path.join(base, "results"),
            "checkpoints": os.path.join(base, "checkpoints"),
            "logs": os.path.join(base, "logs")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def build_models(task_info, method_kwargs, system, device):
    x_dim, u_dim = system.x_dim, system.u_dim
    model_dir = os.path.join(REPO_ROOT, task_info["model_dir"])

    h_source = method_kwargs.get("h_source", task_info.get("h_source", "learned"))
    if h_source == "analytical":
        h_model = system.create_constraint_model().float().to(device)
    elif h_source == "learned":
        h_model = load_mlp(os.path.join(model_dir, "h.pth"), x_dim, 1, device)
    else:
        raise ValueError(f"unknown h_source: {h_source!r}")

    f_pi_source = method_kwargs.get("f_pi_source", task_info["f_pi_source"])
    if f_pi_source == "analytical":
        f_pi_model = system.create_closed_loop_dynamics_model().float().to(device)
    elif f_pi_source == "learned":
        data_npz = np.load(os.path.join(REPO_ROOT, task_info["data_path"]))
        data = {k: data_npz[k].astype(np.float32) for k in data_npz.keys()}
        f_model = load_mlp(os.path.join(model_dir, "f.pth"), x_dim + u_dim, x_dim, device)
        pi_model = system.create_policy().float().to(device)
        f_pi_model = create_closed_loop_dynamics_model(
            f_model, pi_model, data, system.x_low, system.x_high, u_dim
        ).to(device)
    else:
        raise ValueError(f"unknown f_pi_source: {f_pi_source!r}")

    h_model.eval()
    f_pi_model.eval()
    return h_model, f_pi_model


FAST_DEFAULTS = {"split": 8, "workers": 8, "threads": 1, "bounds": "lp"}


def make_verifier(name, milp_threads, milp_opts, fast_opts=None):
    """The verifier handed to the training loop. Both methods receive the same one."""
    if name == "milp":
        solver_kwargs = {"threads": milp_threads, **milp_opts}
        return functools.partial(milp_verify, solver_kwargs=solver_kwargs)
    if name == "milp_fast":
        # training-time verifier; the saved certificate is re-proved by the reference MILP below
        from cersyve.milp import milp_fast_verify
        opts = {**FAST_DEFAULTS, **(fast_opts or {})}
        return functools.partial(milp_fast_verify, **opts)
    raise ValueError(f"unknown verifier: {name!r}")


def run_seed(task, method, seed, task_info, method_kwargs, system, device, args):
    t_run0 = time.perf_counter()
    started = datetime.now().isoformat(timespec="seconds")
    load_start = os.getloadavg()[0]
    dirs = task_dirs(task_info)
    suffix = f"_{args.tag}" if args.tag else ""
    label = f"{method}{suffix}"
    out_dir = os.path.join(args.out_dir or dirs["results"], label)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(dirs["checkpoints"], f"V_finetune_{label}_seed{seed}.pth")
    log_dir = dirs["logs"] + os.sep

    h_model, f_pi_model = build_models(task_info, method_kwargs, system, device)
    v_pretrain_path = os.path.join(dirs["base"], args.v_pretrain)
    V_model = load_mlp(v_pretrain_path, system.x_dim, 1, device)

    kwargs = {k: v for k, v in method_kwargs.items() if k not in RUNNER_KEYS}
    milp_opts = {k[len("milp__"):]: kwargs.pop(k)
                 for k in [k for k in kwargs if k.startswith("milp__")]}
    fast_opts = {k[len("milp_fast__"):]: kwargs.pop(k)
                 for k in [k for k in kwargs if k.startswith("milp_fast__")]}
    verifier_name = method_kwargs.get("verifier", args.verifier)
    verifier = make_verifier(verifier_name, args.milp_threads, milp_opts, fast_opts)

    trainer = TRAINERS[method]
    log_path, tfr_train, iterations, stats = trainer(
        V_model=V_model, f_pi_model=f_pi_model, h_model=h_model,
        x_low=system.x_low, x_high=system.x_high,
        log_dir=log_dir, save_path=save_path, device=device, seed=seed,
        verifier=verifier, **kwargs,
    )

    # ---- final reference re-verification (both methods, always) -----------------
    V_saved = load_mlp(save_path, system.x_dim, 1, "cpu")
    h_cpu, f_cpu = build_models(task_info, method_kwargs, system, "cpu")
    print("Reference re-verification of the saved certificate (cersyve.milp.milp_verify)...")
    ref_solves = []
    load_ref = os.getloadavg()[0]
    t0 = time.perf_counter()
    ok, ref_ce = milp_verify(V_saved, h_cpu, f_cpu, system.x_low, system.x_high,
                        timeout=args.reverify_timeout,
                        solver_kwargs={"threads": args.milp_threads, **milp_opts},
                        record=ref_solves)
    ref_seconds = time.perf_counter() - t0
    reference = {"result": ok, "verified": ok is True, "seconds": ref_seconds,
                 "timeout": args.reverify_timeout, "solves": ref_solves}
    trained_proof = bool(stats.get("formally_verified", False))
    verified = trained_proof and ok is True
    if trained_proof and ok is not True:
        print(f"REFERENCE RE-VERIFICATION DID NOT CONFIRM THE TRAINING PROOF ({ok}): "
              f"reported as unverified")
    elif (not trained_proof) and ok is True:
        print("Note: the training loop stopped without a proof, but the final weights "
              "verify under the reference MILP; the run still counts as unverified "
              "(the loop's own stopping rule did not certify it).")

    # ---- TFR: the project's definition, one fixed evaluation draw for every method ----
    V_dev = load_mlp(save_path, system.x_dim, 1, device)
    rng_state = np.random.get_state()
    np.random.seed(args.eval_seed)
    tfr = estimate_tfr(V_dev, h_model, f_pi_model, system.x_low, system.x_high,
                       n_samples=args.eval_samples, horizon=100, device=device)
    np.random.set_state(rng_state)

    verifier_log = list(stats.get("verifier_log", []))
    verifier_log.append({"call": len(verifier_log) + 1, "iter": iterations, "stage": "reference",
                         "verdict": {True: "proof", False: "refutation", None: "inconclusive"}[ok],
                         "seconds": ref_seconds, "load1_at_call": load_ref,
                         "n_counterexamples": 0 if ref_ce is None else int(ref_ce.shape[1]),
                         "solves": ref_solves, "checkpoint": save_path})
    stages = {}
    for c in verifier_log:
        s = stages.setdefault(c["stage"], {"calls": 0, "solves": 0, "seconds": 0.0,
                                           "proof": 0, "refutation": 0, "inconclusive": 0})
        s["calls"] += 1
        s["solves"] += len(c["solves"])
        s["seconds"] += c["seconds"]
        s[c["verdict"]] += 1

    result = {
        "task": task,
        "method": method,
        "tag": args.tag,
        "seed": seed,
        "verified": verified,
        "training_proof": trained_proof,
        "stalled": bool(stats.get("stalled", False)),
        "tfr": tfr,                                   # measured on the final weights
        "tfr_certified": tfr if verified else 0.0,    # the table's number: hard zero if unverified
        "tfr_train_estimate": float(tfr_train),       # the loop's own 100k-state estimate
        "tfr_eval": {"n_samples": args.eval_samples, "horizon": 100, "seed": args.eval_seed},
        "iterations": iterations,
        "iterations_definition": "0-based index of the training-loop iteration in which the "
                                 "loop stopped (the proof iteration, or max_iter-1)",
        "grad_steps": stats.get("grad_steps"),
        "region_sweep_seconds": stats.get("region_sweep_seconds"),
        "region_sweeps": stats.get("region_sweeps"),
        "repair_batch_points": stats.get("repair_batch_points"),
        "rank_preprocessing_seconds": stats.get("rank_preprocessing_seconds"),
        "final_proof_s": stats.get("final_proof_s"),   # seconds of the loop's proving call
        "final_proof_s_definition": "wall seconds of the training-loop verifier call that proved "
                                    "the certificate (the baseline paper's verification time)",
        "verifier_calls": stats["verifier_calls"] + 1,
        "verifier_calls_training": stats["verifier_calls"],
        "milp_solves": stats["milp_solves"] + len(ref_solves),
        "milp_seconds": stats["milp_seconds"] + ref_seconds,
        "milp_seconds_training": stats["milp_seconds"],
        "verifier_proofs": stats.get("verifier_proofs"),
        "verifier_refutations": stats.get("verifier_refutations"),
        "verifier_inconclusive": stats.get("verifier_inconclusive"),
        "verifier_proofs_total": sum(c["verdict"] == "proof" for c in verifier_log),
        "verifier_refutations_total": sum(c["verdict"] == "refutation" for c in verifier_log),
        "verifier_inconclusive_total": sum(c["verdict"] == "inconclusive" for c in verifier_log),
        "verifier_stages": stages,
        "verifier_log": verifier_log,
        "reference_reverify": reference,
        "wall_time_s": stats.get("wall_time_s"),
        "verified_wall_s": stats.get("verified_wall_s"),
        "certified_wall_s": (None if stats.get("verified_wall_s") is None or not verified
                             else stats["verified_wall_s"] + ref_seconds),
        "end_to_end_wall_s": time.perf_counter() - t_run0,
        "load1_start": load_start,
        "load1_end": os.getloadavg()[0],
        "n_cpus": os.cpu_count(),
        "verifier": verifier_name,
        "milp_threads": args.milp_threads,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "device": device,
        "host": socket.gethostname(),
        "started_at": started,
        "code_sha": git_sha(),
        "code_root": REPO_ROOT,
        "v_pretrain": v_pretrain_path,
        "log_path": log_path,
        "model_path": save_path,
        "config": method_kwargs,
    }
    with open(os.path.join(out_dir, f"seed{seed}.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result, out_dir


def _stat(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "n": len(vals)}


def aggregate(results, out_dir):
    """aggregate.json over every seed JSON present in out_dir (seeds may be run as
    separate processes), with the sample standard deviation."""
    import glob
    by_seed = {}
    for pth in sorted(glob.glob(os.path.join(out_dir, "seed*.json"))):
        try:
            with open(pth) as f:
                r = json.load(f)
            by_seed[r["seed"]] = r
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    for r in results:
        by_seed[r["seed"]] = r
    results = [by_seed[k] for k in sorted(by_seed, key=lambda k: ([1, 0, 2].index(k) if k in (1, 0, 2) else 9, k))]
    fields = ["tfr", "tfr_certified", "iterations", "verifier_calls", "milp_solves",
              "milp_seconds", "wall_time_s", "certified_wall_s", "end_to_end_wall_s"]
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seeds": [r["seed"] for r in results],
        "n_verified": sum(bool(r["verified"]) for r in results),
        "n_runs": len(results),
        "std_definition": "sample standard deviation (ddof=1) over the seeds listed",
        "stats": {f: _stat([r.get(f) for r in results]) for f in fields},
        "per_seed": [{k: r.get(k) for k in ["seed", "verified", "tfr", "tfr_certified",
                                             "iterations", "verifier_calls", "milp_solves",
                                             "milp_seconds", "wall_time_s", "certified_wall_s",
                                             "end_to_end_wall_s"]} for r in results],
    }
    with open(os.path.join(out_dir, "aggregate.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (default: the task's configured seeds)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default=None,
                        help="Result root (default: models/<task>/results)")
    parser.add_argument("--config", default=None, metavar="JSON",
                        help="JSON file of loop parameters applied as overrides (configs/final/<row>.json)")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value (repeatable)")
    parser.add_argument("--tag", default="",
                        help="Suffix for result dirs and checkpoint names (variant runs)")
    parser.add_argument("--v-pretrain", default="V_pretrain.pth",
                        help="Pretrained value checkpoint filename in the task's model dir")
    parser.add_argument("--verifier", default="milp", choices=["milp", "milp_fast"],
                        help="Verifier used inside the training loop (same for both methods)")
    parser.add_argument("--milp-threads", type=int, default=8,
                        help="Gurobi threads per MILP solve")
    parser.add_argument("--reverify-timeout", type=int, default=1800,
                        help="Time limit (s) per solve of the final reference re-verification")
    parser.add_argument("--eval-samples", type=int, default=1_000_000)
    parser.add_argument("--eval-seed", type=int, default=12345)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.setdefault("CERSYVE_MILP_THREADS", str(args.milp_threads))

    overrides = {}
    if args.config:
        with open(args.config) as fh:
            overrides.update(json.load(fh))
    overrides.update(parse_overrides(args.override))
    cfg = get_config(args.task, args.method, overrides)
    task_info, method_kwargs = cfg["task"], cfg["method_kwargs"]
    system = resolve_system(task_info["system"])
    seeds = args.seeds if args.seeds is not None else task_info["seeds"]

    print(f"Task: {args.task} | Method: {args.method} | Seeds: {seeds} | Device: {device}")
    print(f"Code: {REPO_ROOT} @ {git_sha()} | cersyve: {cersyve.__file__}")
    print(f"Verifier: {args.verifier}, {args.milp_threads} solver threads; "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    results = []
    out_dir = None
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        r, out_dir = run_seed(args.task, args.method, seed, task_info, method_kwargs,
                              system, device, args)
        results.append(r)
        print(f"Seed {seed}: verified={r['verified']} TFR={r['tfr']:.4f} "
              f"iterations={r['iterations']} verifier_calls={r['verifier_calls']} "
              f"milp_solves={r['milp_solves']} milp_seconds={r['milp_seconds']:.1f} "
              f"wall={r['wall_time_s']:.0f}s")

    summary = aggregate(results, out_dir)
    s = summary["stats"]
    print(f"\nVerified runs: {summary['n_verified']}/{summary['n_runs']}")
    print(f"TFR (certified): {100*s['tfr_certified']['mean']:.2f} ± {100*s['tfr_certified']['std']:.2f}")
    if s["iterations"]:
        print(f"Iterations: {s['iterations']['mean']:.1f} ± {s['iterations']['std']:.1f}")
    print(f"Verifier calls: {s['verifier_calls']['mean']:.2f} ± {s['verifier_calls']['std']:.2f}; "
          f"MILP seconds: {s['milp_seconds']['mean']:.1f} ± {s['milp_seconds']['std']:.1f}")
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
