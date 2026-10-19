"""Write a CARBS spec for one benchmark and one method.

    python scripts/make_spec.py TASK {pgd_milp,enum_milp} [--max-trials 60] [--parallel 4] [--trial-timeout 7200] [--max-iter N]

The search space is centred at the task's configuration (configs/<task>.py). Writes
models/<task>/campaign/specs/<task>_<method>.json for examples/campaign.py.
"""

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from configs import get_config  # noqa: E402

# name, scale, lo, hi, integer
ENUM_SPACE = [
    ("max_skip", "log", 50, 6000, True),
    ("enum_freq", "log", 100, 5000, True),
    ("enum_n_samples", "log", 2000, 100000, True),
    ("enum_coverage", "linear", 30.0, 99.0, False),
    ("enum_n_trees", "log", 50, 3000, True),
    ("refill_threshold", "log", 50, 5000, True),
    ("reg_coef", "log", 0.01, 1.0, False),
    ("eps_h", "log", 0.001, 0.1, False),
    ("eps_v", "log", 0.001, 0.1, False),
    ("lr", "log", 1e-5, 1e-3, False),
    ("tol", "log", 1e-4, 2e-2, False),
    ("fallback_radius", "log", 0.005, 0.3, False),
    # fallback sampling knobs: they decide whether the dry-spell trigger can fire
    ("fallback_eval_samples", "log", 10000, 100000, True),
    ("fallback_eval_rounds", "linear", 1, 5, True),
]

PGD_SPACE = [
    ("max_skip", "log", 50, 6000, True),
    ("search_size", "log", 100, 5000, True),
    ("pgd_step", "linear", 5, 60, True),
    ("pgd_eps", "log", 0.01, 0.3, False),
    ("bnd_eps", "log", 0.01, 0.5, False),
    ("reg_coef", "log", 0.01, 1.0, False),
    ("eps_h", "log", 0.001, 0.1, False),
    ("eps_v", "log", 0.001, 0.1, False),
    ("lr", "log", 1e-5, 1e-3, False),
    ("tol", "log", 1e-4, 2e-2, False),
]

SPACES = {"pgd_milp": PGD_SPACE, "enum_milp": ENUM_SPACE}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task")
    ap.add_argument("method", choices=list(SPACES))
    ap.add_argument("--max-trials", type=int, default=60)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--trial-timeout", type=int, default=7200)
    ap.add_argument("--max-iter", type=int, default=None)
    args = ap.parse_args()

    cfg = get_config(args.task, args.method, {})
    kwargs = cfg["method_kwargs"]
    search_space = []
    for name, scale, lo, hi, integer in SPACES[args.method]:
        centre = min(max(kwargs[name], lo), hi)
        search_space.append({"name": name, "scale": scale, "lo": lo, "hi": hi,
                             "center": centre, "integer": integer})
    name = f"{args.task}_{args.method}"
    spec = {
        "task": args.task, "method": args.method, "device": "cuda",
        "search_seed": 1, "confirm_seeds": [0, 2],
        "tag_prefix": name[:2] + "_",
        "max_calls": None,
        "cost": "final_proof_s",
        "win": {"tfr_min_pct": 0.0},
        "strong_win": {"tfr_min_pct": 0.0},
        "stop_on": "none", "observe_confirmation_mean": True,
        "max_trials": args.max_trials, "parallel": args.parallel,
        "trial_timeout_s": args.trial_timeout, "max_retries": 1,
        "fixed": {"max_iter": args.max_iter or kwargs["max_iter"]},
        "search_space": search_space,
        "gpus": [0],
    }
    root = os.path.join(REPO, cfg["task"]["model_dir"], "campaign", "specs")
    os.makedirs(root, exist_ok=True)
    glob_path = os.path.join(root, "_global.json")
    if not os.path.exists(glob_path):
        with open(glob_path, "w") as f:
            json.dump({
                "python": sys.executable,
                "solver": "GUROBI", "global_slots": args.parallel,
                "trial_env": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                "runner_args": ["--milp-threads", "8", "--verifier", "milp"],
                "license_error_regex": ["GurobiError", "Error code \\d{5}",
                                        "[Ll]icense (?:limit|expired|error|server)",
                                        "[Tt]oo many (?:sessions|clients|users)",
                                        "Unable to (?:connect|retrieve).*license"],
                "n_gpus": 1, "gpu_offset": 0,
            }, f, indent=2)
    out = os.path.join(root, f"{name}.json")
    with open(out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"written {out}")


if __name__ == "__main__":
    main()
