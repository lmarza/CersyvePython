"""Re-solve the logged verifier calls of finished runs on a quiet machine.

    python scripts/replay_verifier.py models/<task>/results/<run>/seed*.json [--only-final] [--threads 8] [--timeout 1800]

Writes <seed>.replay.json next to each result: per-call replay seconds, the replayed
verdicts (which must match) and final_proof_s_replay.
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "examples"))

import run_finetune as rf                      # noqa: E402
from configs import get_config                 # noqa: E402
from cersyve import load_mlp                   # noqa: E402
from cersyve.milp import milp_verify           # noqa: E402


def _abs(p):
    """Paths in the result records are relative to the repository root."""
    return p if not p or os.path.isabs(p) else os.path.join(REPO, p)


def replay(path, threads, timeout, only_final=False):
    d = json.load(open(path))
    cfg = get_config(d["task"], d["method"], {})
    ti, mk = cfg["task"], d.get("config") or cfg["method_kwargs"]
    system = rf.resolve_system(ti["system"])
    h, f = rf.build_models(ti, mk, system, "cpu")
    out = {"source": path, "threads": threads, "timeout": timeout, "calls": [], "mismatch": 0}
    total = 0.0
    n_train = sum(1 for c in d["verifier_log"] if c["stage"] != "reference")
    calls = d["verifier_log"]
    if only_final:
        # paper convention: the final proving call of the loop and the reference re-check
        proofs = [c for c in calls if c["stage"] != "reference" and c["verdict"] == "proof"]
        calls = (proofs[-1:] if proofs else []) + [c for c in calls if c["stage"] == "reference"]
    for c in calls:
        ckpt = _abs(c.get("checkpoint"))
        if not ckpt or not os.path.exists(ckpt):
            # runs made before per-call checkpoints existed: the PGD loop saved
            # before_milp_iter<i+1>.pth, and a proof at the last training call
            # verified exactly the weights that were saved as the final model
            cand = os.path.join(_abs(d.get("log_path", "")), "milp_checkpoints",
                                f"before_milp_iter{int(c['iter']) + 1}.pth")
            if os.path.exists(cand):
                ckpt = cand
            elif (c["stage"] == "reference" or (c["call"] == n_train and d.get("training_proof"))) \
                    and d.get("model_path") and os.path.exists(_abs(d["model_path"])):
                ckpt = _abs(d["model_path"])
        if not ckpt or not os.path.exists(ckpt):
            out["calls"].append({**{k: c[k] for k in ("call", "iter", "stage", "verdict")},
                                 "replay": None, "note": "no checkpoint"})
            continue
        V = load_mlp(ckpt, system.x_dim, 1, "cpu")
        rec = []
        load1 = os.getloadavg()[0]
        t0 = time.perf_counter()
        ok, _ = milp_verify(V, h, f, system.x_low, system.x_high, timeout=timeout, verbose=False,
                            solver_kwargs={"threads": threads}, record=rec)
        dt = time.perf_counter() - t0
        verdict = {True: "proof", False: "refutation", None: "inconclusive"}[ok]
        if verdict != c["verdict"]:
            out["mismatch"] += 1
        total += dt
        out["calls"].append({"call": c["call"], "iter": c["iter"], "stage": c["stage"],
                             "verdict": c["verdict"], "verdict_replay": verdict,
                             "seconds_run": c["seconds"], "load1_run": c.get("load1_at_call"),
                             "seconds_replay": dt, "load1_replay": load1,
                             "solves_replay": rec})
    out["milp_seconds_run"] = d["milp_seconds"]
    out["milp_seconds_replay"] = total
    proofs = [c for c in out["calls"] if c.get("stage") != "reference" and c.get("verdict_replay") == "proof"]
    out["final_proof_s_replay"] = proofs[-1]["seconds_replay"] if proofs else None
    out["n_replayed"] = sum(1 for c in out["calls"] if c.get("replay", 1) is not None)
    dst = path.replace(".json", ".replay.json")
    json.dump(out, open(dst, "w"), indent=2, default=str)
    print(f"{path}: run {d['milp_seconds']:.1f} s -> replay {total:.1f} s over {out['n_replayed']} calls, "
          f"mismatched verdicts {out['mismatch']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--only-final", action="store_true", help="replay only the final proof and the reference re-check")
    args = ap.parse_args()
    for p in args.results:
        replay(p, args.threads, args.timeout, only_final=args.only_final)


if __name__ == "__main__":
    main()
