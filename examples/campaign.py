"""CARBS-guided campaign driver: one process per spec, searching the training-loop
hyperparameters of one method through examples/run_finetune.py subprocess trials,
with warm start, confirmation seeds and win detection.

    python examples/campaign.py --root models/<task>/campaign --task <spec>
    python examples/campaign.py --root ... --task <spec> --check   # validate only

Layout under --root:  specs/_global.json, specs/<spec>.json, slots/, state/<spec>/
(ledger.jsonl, status.json, carbs.pkl, trials/*.log, DONE), wins/<task>/<tag>/.
Trials run THIS clone's examples/run_finetune.py and write their result JSONs to
models/<task>/results/<method>_<tag>/seed<k>.json.

Spec fields (see configs for the parameter names; only training-loop knobs may be
searched, never tol / milp_timeout / the verifier):

    task, method, device, search_seed, confirm_seeds, search_space, fixed,
    max_calls              optional: a trial with more verifier calls counts as infeasible
    cost                   "certified_wall_s" (default) | "milp_seconds" | "wall_time_s"
    win / strong_win       {tfr_min_pct}: certified mean TFR a confirmed candidate must exceed
    max_trials, parallel, trial_timeout_s, max_retries, stop_on, warm_start

Observation for CARBS: output = 100 * tfr_certified when the trial is verified and
within the screening call budget, else 0; cost = the trial's `cost` field
(seconds; a failed trial costs its wall time). Final acceptance is evaluated on
the unrounded means over the search seed plus the confirmation seeds.
"""

import argparse
import fcntl
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

SUGGESTION_ID_KEY = "suggestion_uuid"
POLL_INTERVAL_S = 2.0
KILL_GRACE_S = 60.0

# tqdm's own progress-bar redraws (not tqdm.write() messages), so a license
# regex never fires on bar noise carried in the trial log.
TQDM_BAR_RE = re.compile(r"\d{1,3}%\|.{0,60}\|\s*\d+/\d+")


# ─────────────────────────────────────────────────────────────────────────────
# Pure logic (importable; no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def trial_feasible(result, task_spec):
    """A trial counts when its certificate is proven (and within max_calls, if set)."""
    max_calls = task_spec.get("max_calls")
    return (bool(result.get("verified"))
            and (max_calls is None or (result.get("verifier_calls") is not None
                                       and result["verifier_calls"] <= float(max_calls))))


def trial_cost(result, task_spec):
    field = task_spec.get("cost", "certified_wall_s")
    value = result.get(field)
    if value is None:
        value = result.get("wall_time_s") or result.get("end_to_end_wall_s") or 1.0
    return max(float(value), 1.0)


def observation_for_result(result, task_spec):
    """Returns (output, cost, feasible) for one finished trial.

    An infeasible trial is charged its full wall time, never its verifier seconds:
    with cost = MILP seconds, the robot arm campaign of 2026-10-08 drifted to
    configurations that never call the verifier (0 calls, 2 s of MILP, output 0),
    because those were the cheapest points CARBS could see."""
    feasible = trial_feasible(result, task_spec)
    if not feasible:
        wall = result.get("end_to_end_wall_s") or result.get("wall_time_s") or 1.0
        return 0.0, max(float(wall), trial_cost(result, task_spec)), False
    output = 100.0 * float(result.get("tfr_certified") or 0.0)
    return output, trial_cost(result, task_spec), True


def seed_win(result, task_spec, win_params):
    """Does one seed look like a candidate worth confirming?"""
    if not trial_feasible(result, task_spec):
        return False
    return 100.0 * result["tfr_certified"] > float(win_params.get("tfr_min_pct", 0.0))


def _mean_std(values):
    vals = [float(v) for v in values]
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def aggregate_confirmation(seed_results, task_spec, win_params, strong_params):
    """Final acceptance on unrounded means over all seeds (search + confirm)."""
    n_verified = sum(bool(r.get("verified")) for r in seed_results)
    per_seed = [{"seed": r.get("seed"), "verified": bool(r.get("verified")),
                 "tfr_pct": 100.0 * float(r.get("tfr") or 0.0),
                 "tfr_certified_pct": 100.0 * float(r.get("tfr_certified") or 0.0),
                 "iterations": r.get("iterations"), "verifier_calls": r.get("verifier_calls"),
                 "milp_solves": r.get("milp_solves"), "milp_seconds": r.get("milp_seconds"),
                 "wall_time_s": r.get("wall_time_s"), "certified_wall_s": r.get("certified_wall_s")}
                for r in seed_results]
    means = {}
    stds = {}
    for key, field in (("tfr", "tfr_certified_pct"), ("calls", "verifier_calls"),
                       ("milp_seconds", "milp_seconds"), ("wall", "wall_time_s"),
                       ("iter", "iterations")):
        vals = [p[field] for p in per_seed if p[field] is not None]
        if vals:
            means[key], stds[key] = _mean_std(vals)
    all_verified = n_verified == len(seed_results)

    def wins(params):
        if not all_verified:
            return False
        max_calls = task_spec.get("max_calls")
        return (means.get("tfr", 0.0) > float(params.get("tfr_min_pct", 0.0))
                and (max_calls is None or means.get("calls", float("inf")) <= float(max_calls)))

    return {
        "confirmed": True, "all_verified": all_verified, "n_verified": n_verified,
        "confirmed_win": wins(win_params), "confirmed_strong": wins(strong_params),
        "per_seed": per_seed, "means": means, "stds": stds,
    }


def confirmation_mean_observation(seed_results, task_spec):
    outputs, costs = [], []
    for r in seed_results:
        o, c, _ = observation_for_result(r, task_spec)
        outputs.append(o)
        costs.append(c)
    return float(np.mean(outputs)), float(np.mean(costs))


def _split_log_lines(text: str):
    return re.split(r"[\r\n]+", text)


def classify_log_text(log_text: str, license_regexes):
    """Scan non-tqdm-bar lines for a traceback or a license error."""
    compiled = [re.compile(p) for p in license_regexes]
    has_traceback = False
    license_line = None
    for line in _split_log_lines(log_text):
        if not line.strip() or TQDM_BAR_RE.search(line):
            continue
        if line.startswith("Traceback (most recent call last)"):
            has_traceback = True
        if license_line is None:
            for pat in compiled:
                if pat.search(line):
                    license_line = line.strip()
                    break
    return {"has_traceback": has_traceback, "license_line": license_line}


def classify_finish(json_exists: bool, log_text: str, license_regexes):
    if json_exists:
        return "ok", None
    info = classify_log_text(log_text, license_regexes)
    if info["license_line"]:
        return "license", info["license_line"]
    if info["has_traceback"]:
        return "crash", "traceback"
    return "crash", "no_json_no_traceback"


def warm_start_matches(config: dict, require_config: dict, search_space):
    for k, v in require_config.items():
        if config.get(k) != v:
            return False, "require_config_mismatch"
    for p in search_space:
        name = p["name"]
        if name not in config:
            return False, "missing_param"
        v = config[name]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, "non_numeric_param"
        if not (p["lo"] <= v <= p["hi"]):
            return False, "out_of_bounds"
    return True, None


def load_warm_start_observations(paths, require_config, search_space, task_spec, impute=None):
    observations = []
    skipped = {}
    loaded = 0
    for path in paths:
        try:
            with open(path) as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError):
            skipped["parse_error"] = skipped.get("parse_error", 0) + 1
            continue
        config = {**(impute or {}), **(result.get("config") or {})}
        ok, reason = warm_start_matches(config, require_config, search_space)
        if not ok:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        output, cost, _ = observation_for_result(result, task_spec)
        input_params = {p["name"]: config[p["name"]] for p in search_space}
        observations.append({"input": input_params, "output": output, "cost": cost,
                              "source": path})
        loaded += 1
    return observations, loaded, skipped


def replay_ledger(events, max_retries=0):
    launched = {}
    search_tags = set()
    search_observed = 0
    confirmations = []
    done = None
    last_attempt = {}
    last_launch_kind = {}
    gave_up = set()
    observed_tags = set()
    for ev in events:
        kind = ev.get("event")
        if kind == "launched":
            key = (ev["tag"], ev["seed"])
            launched[key] = ev
            gave_up.discard(key)
            last_attempt[key] = ev.get("attempt", 1)
            last_launch_kind[key] = ev.get("kind", "search")
            if ev.get("kind") == "search":
                search_tags.add(ev["tag"])
        elif kind == "finished":
            key = (ev["tag"], ev["seed"])
            launched.pop(key, None)
            outcome = ev.get("outcome")
            attempt = last_attempt.get(key, 1)
            lkind = last_launch_kind.get(key)
            if lkind == "confirm" and outcome not in (None, "ok"):
                if outcome == "timeout" or attempt > max_retries:
                    gave_up.add(key)
        elif kind == "observed":
            if ev.get("kind") != "confirmation_mean":
                search_observed += 1
                observed_tags.add(ev["tag"])
        elif kind == "confirmation":
            confirmations.append(ev)
        elif kind == "done":
            done = ev
    return {
        "inflight": launched,
        "search_launched": len(search_tags),
        "search_observed": search_observed,
        "confirmations": confirmations,
        "done": done,
        "gave_up": gave_up,
        "observed_tags": observed_tags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json_atomic(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_ledger(path):
    events = []
    if not os.path.exists(path):
        return events
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def append_ledger(path, event):
    event = dict(event)
    event.setdefault("time", datetime.now().isoformat(timespec="seconds"))
    with open(path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ─────────────────────────────────────────────────────────────────────────────
# CARBS setup
# ─────────────────────────────────────────────────────────────────────────────

def build_search_params(search_space):
    from carbs import Param, LogSpace, LinearSpace
    params = []
    for p in search_space:
        cls = LogSpace if p["scale"] == "log" else LinearSpace
        space = cls(min=p["lo"], max=p["hi"], is_integer=bool(p.get("integer", False)))
        params.append(Param(p["name"], space, p["center"]))
    return params


def build_carbs_params(task_spec):
    from carbs import CARBSParams
    kwargs = dict(task_spec.get("carbs_params") or {})
    kwargs.setdefault("better_direction_sign", 1)
    kwargs.setdefault("is_wandb_logging_enabled", False)
    kwargs.setdefault("is_saved_on_every_observation", False)
    return CARBSParams(**kwargs)


# Parameters that define the certification condition or the verifier: never searched.
# (tol is the training-side dilation of the violation set used by filter_counterexample
# and the enumeration labels; the MILP checks V <= 0 and h > 0 / V(f(x)) > 0 with
# VERIFY_MARGIN 0 regardless of tol, so tol may be searched.)
PROTECTED_KEYS = {"milp_timeout", "verifier", "reg_method", "search_method"}


def verify_config_keys(task_spec):
    """Fail fast if configs.get_config rejects any search_space/fixed key, or if a
    protected key is searched."""
    from configs import get_config, TASKS, METHODS
    task = task_spec["task"]
    method = task_spec.get("method", "enum_milp")
    if task not in TASKS:
        raise SystemExit(f"campaign spec error: unknown task {task!r}; expected one of {TASKS}")
    if method not in METHODS:
        raise SystemExit(f"campaign spec error: unknown method {method!r}; expected one of {list(METHODS)}")
    bad = [p["name"] for p in task_spec["search_space"] if p["name"] in PROTECTED_KEYS]
    if bad:
        raise SystemExit(f"campaign spec error: {bad} change the certification condition and must not be searched")
    probe = {p["name"]: p["center"] for p in task_spec["search_space"]}
    probe.update(task_spec.get("fixed") or {})
    try:
        get_config(task, method, probe)
    except (KeyError, ValueError) as e:
        raise SystemExit(f"campaign spec key validation failed for task {task!r}: {e}")


def find_warm_start_sources(code_dir, patterns):
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(os.path.join(code_dir, pat), recursive=True)))
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Slots (global concurrency cap, crash-safe via flock)
# ─────────────────────────────────────────────────────────────────────────────

def acquire_slot(root, global_slots):
    slot_dir = os.path.join(root, "slots")
    os.makedirs(slot_dir, exist_ok=True)
    for k in range(global_slots):
        path = os.path.join(slot_dir, f"slot{k}.lock")
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return k, fd
        except OSError:
            os.close(fd)
    return None, None


def release_slot(fd):
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


# ─────────────────────────────────────────────────────────────────────────────
# Trial launching
# ─────────────────────────────────────────────────────────────────────────────

def format_override_value(value):
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return repr(value)
    return str(value)


def build_overrides(suggestion, search_space, fixed):
    int_names = {p["name"] for p in search_space if p.get("integer")}
    overrides = {}
    for k, v in suggestion.items():
        if k == SUGGESTION_ID_KEY:
            continue
        if k in int_names:
            v = int(round(v))
        overrides[k] = v
    overrides.update(fixed)
    return overrides


def build_trial_cmd(python, task, method, tag, seed, device, overrides, v_pretrain,
                    runner_args=None):
    cmd = [python, "examples/run_finetune.py",
           "--task", task, "--method", method,
           "--seeds", str(seed), "--device", device,
           "--tag", tag]
    if v_pretrain:
        cmd += ["--v-pretrain", str(v_pretrain)]
    for k, v in sorted(overrides.items()):
        cmd += ["--override", f"{k}={format_override_value(v)}"]
    cmd += list(runner_args or [])
    return cmd


def default_launcher(cmd, cwd, env, log_path, extra_fds):
    logf = open(log_path, "ab")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                             pass_fds=tuple(extra_fds), start_new_session=True)
    logf.close()
    return proc.pid, proc.pid, proc


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

class Trial:
    def __init__(self, tag, seed, kind, attempt, overrides, suggestion,
                 start_time, pid, pgid, slot_fd, max_iter, log_path, proc=None):
        self.tag = tag
        self.seed = seed
        self.kind = kind  # "search" | "confirm"
        self.attempt = attempt
        self.overrides = overrides
        self.suggestion = suggestion
        self.start_time = start_time
        self.pid = pid
        self.pgid = pgid
        self.slot_fd = slot_fd
        self.max_iter = max_iter
        self.log_path = log_path
        self.proc = proc


def _pid_matches(pid, tag, task):
    cmdline_path = f"/proc/{pid}/cmdline"
    if not os.path.exists(cmdline_path):
        return False
    try:
        with open(cmdline_path, "rb") as f:
            raw = f.read()
    except OSError:
        return False
    parts = raw.split(b"\x00")
    text = " ".join(p.decode(errors="replace") for p in parts)
    return tag in text and task in text


class Driver:
    def __init__(self, root, spec_name, launcher=default_launcher, poll_interval=POLL_INTERVAL_S):
        self.root = os.path.abspath(root)
        self.spec_name = spec_name
        self.launcher = launcher
        self.poll_interval = poll_interval

        self.global_spec = read_json(os.path.join(self.root, "specs", "_global.json"))
        self.task_spec = read_json(os.path.join(self.root, "specs", f"{spec_name}.json"))
        self.task = self.task_spec["task"]
        self.method = self.task_spec.get("method", "enum_milp")

        self.state_dir = os.path.join(self.root, "state", spec_name)
        self.trials_dir = os.path.join(self.state_dir, "trials")
        os.makedirs(self.trials_dir, exist_ok=True)
        self.ledger_path = os.path.join(self.state_dir, "ledger.jsonl")
        self.status_path = os.path.join(self.state_dir, "status.json")
        self.carbs_pkl = os.path.join(self.state_dir, "carbs.pkl")
        self.done_path = os.path.join(self.state_dir, "DONE")

        from configs import get_config
        model_dir = get_config(self.task, self.method, {})["task"]["model_dir"]
        self.results_root = os.path.join(CODE_DIR, model_dir, "results")
        self.wins_dir = os.path.join(self.root, "wins", self.task)

        self.carbs = None
        self.search_index = 0
        self.search_observed = 0
        self.verified_count = 0
        self.feasible_count = 0
        self.best_feasible = None  # {"tfr_pct", "cost", "calls", "tag"}
        self.confirmations = []
        self.warm_start_info = None
        self.inflight = {}
        self.last_error = None
        self.pending_confirm_tags = set()
        self.pending_retries = []
        self.gave_up = set()
        self.launch_index = {}

    # ---- specs / carbs bootstrap ----

    def verify(self):
        verify_config_keys(self.task_spec)

    def _build_fresh_carbs(self):
        from carbs import CARBS
        params = build_search_params(self.task_spec["search_space"])
        carbs_params = build_carbs_params(self.task_spec)
        self.carbs = CARBS(carbs_params, params)

    def _save_carbs(self):
        import pickle
        tmp = f"{self.carbs_pkl}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump((self.carbs, self.search_index), f)
        os.replace(tmp, self.carbs_pkl)

    def _load_carbs(self):
        import pickle
        with open(self.carbs_pkl, "rb") as f:
            self.carbs, self.search_index = pickle.load(f)

    def do_warm_start(self):
        ws = self.task_spec.get("warm_start") or {}
        if not ws.get("enabled"):
            self.warm_start_info = {"enabled": False, "loaded": 0, "skipped": {}}
            return
        from carbs import ObservationInParam
        paths = find_warm_start_sources(CODE_DIR, ws.get("sources", []))
        observations, loaded, skipped = load_warm_start_observations(
            paths, ws.get("require_config", {}), self.task_spec["search_space"],
            self.task_spec, impute=ws.get("impute"))
        if observations:
            self.carbs.initialize_from_observations(
                [ObservationInParam(input=dict(o["input"]), output=o["output"], cost=o["cost"])
                 for o in observations])
        self.warm_start_info = {"enabled": True, "candidates": len(paths),
                                 "loaded": loaded, "skipped": skipped}
        append_ledger(self.ledger_path, {"event": "warm_start", "candidates": len(paths),
                                          "loaded": loaded, "skipped": skipped})

    # ---- resume ----

    def resume(self):
        events = read_ledger(self.ledger_path)
        replay = replay_ledger(events, max_retries=self.task_spec.get("max_retries", 0))
        self.search_launched_from_ledger = replay["search_launched"]
        self.search_observed = replay["search_observed"]
        self.gave_up |= replay["gave_up"]
        self._observed_tags_from_ledger = replay["observed_tags"]
        self.confirmations = [
            {"tag": ev["tag"], "confirmed": True,
             "confirmed_win": ev.get("confirmed_win"), "confirmed_strong": ev.get("confirmed_strong"),
             "per_seed": ev.get("per_seed"), "means": ev.get("means"), "stds": ev.get("stds"),
             "reason": ev.get("reason")}
            for ev in replay["confirmations"]
        ]
        self.search_index = max(self.search_index, self.search_launched_from_ledger)
        for ev in events:
            if ev.get("event") == "launched":
                entry = self.launch_index.setdefault(ev["tag"], {})
                if ev.get("suggestion"):
                    entry["suggestion"] = ev["suggestion"]
                entry.setdefault("overrides", ev.get("overrides", {}))
        for (tag, seed), ev in replay["inflight"].items():
            json_path = self._seed_json_path(tag, seed)
            if os.path.exists(json_path):
                self._finish_trial_from_launch_event(ev, json_path)
                continue
            pid = ev.get("pid")
            if pid and _pid_matches(pid, tag, self.task):
                self.inflight[(tag, seed)] = Trial(
                    tag=tag, seed=seed, kind=ev.get("kind", "search"),
                    attempt=ev.get("attempt", 1), overrides=ev.get("overrides", {}),
                    suggestion=ev.get("suggestion"), start_time=ev.get("start_time", time.time()),
                    pid=pid, pgid=ev.get("pgid", pid), slot_fd=None,
                    max_iter=self._resolved_max_iter(ev.get("overrides", {})),
                    log_path=os.path.join(self.trials_dir, f"{tag}_s{seed}.log"), proc=None)
            else:
                self._handle_crash(ev, "resume_orphan")
        still_outstanding = {tag for (tag, seed), ev in replay["inflight"].items()
                              if ev.get("kind") == "search"}
        self._rebuild_search_stats(events, skip_tags=still_outstanding)
        self._rebuild_pending_confirms()
        for tag in list(self.pending_confirm_tags):
            self._maybe_confirm(tag)

    def _note_search_result(self, tag, result):
        output, cost, feasible = observation_for_result(result, self.task_spec)
        if result.get("verified"):
            self.verified_count += 1
        if feasible:
            self.feasible_count += 1
            entry = {"tfr_pct": output, "cost": cost, "calls": result.get("verifier_calls"),
                     "milp_seconds": result.get("milp_seconds"), "wall": result.get("wall_time_s"),
                     "tag": tag}
            if self.best_feasible is None or entry["tfr_pct"] > self.best_feasible["tfr_pct"]:
                self.best_feasible = entry
        return output, cost, feasible

    def _rebuild_search_stats(self, events, skip_tags):
        search_tags = {ev["tag"] for ev in events
                       if ev.get("event") == "launched" and ev.get("kind") == "search"}
        for tag in search_tags - skip_tags:
            p = self._seed_json_path(tag, self.task_spec["search_seed"])
            if not os.path.exists(p):
                continue
            self._note_search_result(tag, read_json(p))

    def _finish_trial_from_launch_event(self, ev, json_path):
        result = read_json(json_path)
        already_observed = ev["tag"] in getattr(self, "_observed_tags_from_ledger", ())
        if ev.get("kind") == "search" and not already_observed:
            self._observe_search_result(ev["tag"], result)
        append_ledger(self.ledger_path, {"event": "finished", "tag": ev["tag"], "seed": ev["seed"],
                                          "outcome": "ok"})

    def _rebuild_pending_confirms(self):
        confirmed_tags = {c["tag"] for c in self.confirmations}
        for tag in list(self.launch_index):
            if tag in confirmed_tags:
                continue
            p = self._seed_json_path(tag, self.task_spec["search_seed"])
            if not os.path.exists(p):
                continue
            if seed_win(read_json(p), self.task_spec, self.task_spec["win"]) and not self._already_confirmed_config(tag):
                self.pending_confirm_tags.add(tag)

    # ---- key paths ----

    def _seed_json_path(self, tag, seed):
        return os.path.join(self.results_root, f"{self.method}_{tag}", f"seed{seed}.json")

    def _tag(self, i):
        return f"{self.task_spec.get('tag_prefix', '')}s{i:04d}"

    # ---- launching ----

    def _env(self):
        env = dict(os.environ)
        env.update(self.global_spec.get("trial_env", {}))
        env["CERSYVE_MILP_SOLVER"] = self.global_spec.get("solver", "GUROBI")
        return env

    def _resolved_max_iter(self, overrides):
        if "max_iter" in overrides:
            return overrides["max_iter"]
        from configs import get_config
        return get_config(self.task, self.method, {})["method_kwargs"]["max_iter"]

    def _launch(self, tag, seed, kind, attempt, overrides, suggestion):
        slot_idx, slot_fd = acquire_slot(self.root, self.global_spec["global_slots"])
        if slot_fd is None:
            return False
        v_pretrain = self.task_spec.get("v_pretrain")
        cmd = build_trial_cmd(self.global_spec["python"], self.task, self.method, tag, seed,
                               self.task_spec["device"], overrides, v_pretrain,
                               self.global_spec.get("runner_args"))
        log_path = os.path.join(self.trials_dir, f"{tag}_s{seed}.log")
        env = self._env()
        n_gpus = int(self.global_spec.get("n_gpus", 0))
        gpus = self.task_spec.get("gpus")
        if not hasattr(self, "_trial_gpu"):
            self._trial_gpu = {}
        if gpus:
            used = [self._trial_gpu.get(k) for k in self.inflight]
            gpu = min(gpus, key=lambda g: used.count(g))
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            self._trial_gpu[(tag, seed)] = gpu
        elif n_gpus > 0 and "CUDA_VISIBLE_DEVICES" not in self.global_spec.get("trial_env", {}):
            off = int(self.global_spec.get("gpu_offset", 0))
            env["CUDA_VISIBLE_DEVICES"] = str((slot_idx + off) % n_gpus)
        try:
            pid, pgid, proc = self.launcher(cmd, CODE_DIR, env, log_path, [slot_fd])
        except Exception as e:
            release_slot(slot_fd)
            self.last_error = f"launch failed for {tag}/seed{seed}: {e}"
            return False
        start_time = time.time()
        trial = Trial(tag=tag, seed=seed, kind=kind, attempt=attempt, overrides=overrides,
                       suggestion=suggestion, start_time=start_time, pid=pid, pgid=pgid,
                       slot_fd=slot_fd, max_iter=self._resolved_max_iter(overrides), log_path=log_path,
                       proc=proc)
        self.inflight[(tag, seed)] = trial
        append_ledger(self.ledger_path, {
            "event": "launched", "tag": tag, "seed": seed, "kind": kind, "attempt": attempt,
            "pid": pid, "pgid": pgid, "overrides": overrides,
            "suggestion": {k: v for k, v in (suggestion or {}).items()},
            "start_time": start_time, "cmd": " ".join(cmd),
        })
        entry = self.launch_index.setdefault(tag, {})
        if suggestion:
            entry["suggestion"] = dict(suggestion)
        entry.setdefault("overrides", overrides)
        return True

    def _launch_search(self):
        suggestion = self.carbs.suggest().suggestion
        overrides = build_overrides(suggestion, self.task_spec["search_space"], self.task_spec.get("fixed", {}))
        tag = self._tag(self.search_index)
        self.search_index += 1
        seed = self.task_spec["search_seed"]
        ok = self._launch(tag, seed, "search", 1, overrides, suggestion)
        if not ok:
            self.carbs.forget_suggestion(suggestion)
            self.search_index -= 1
        return ok

    def _launch_confirm(self, tag, overrides):
        launched_any = False
        for seed in self.task_spec["confirm_seeds"]:
            if (tag, seed) in self.gave_up:
                continue
            if (tag, seed) in self.inflight or os.path.exists(self._seed_json_path(tag, seed)):
                continue
            if self._launch(tag, seed, "confirm", 1, overrides, None):
                launched_any = True
        return launched_any

    # ---- observing ----

    def _observe_search_result(self, tag, result):
        from carbs import ObservationInParam
        output, cost, feasible = self._note_search_result(tag, result)
        suggestion = self._find_suggestion_for_tag(tag)
        self.carbs.observe(ObservationInParam(input=suggestion, output=output, cost=cost))
        self._save_carbs()
        self.search_observed += 1
        append_ledger(self.ledger_path, {"event": "observed", "tag": tag, "output": output,
                                          "cost": cost, "is_failure": False, "feasible": feasible,
                                          "verified": bool(result.get("verified")),
                                          "verifier_calls": result.get("verifier_calls"),
                                          "milp_seconds": result.get("milp_seconds"),
                                          "wall_time_s": result.get("wall_time_s"),
                                          "tfr_pct": 100.0 * float(result.get("tfr") or 0.0)})
        if seed_win(result, self.task_spec, self.task_spec["win"]) and not self._already_confirmed_config(tag):
            self.pending_confirm_tags.add(tag)

    def _find_suggestion_for_tag(self, tag):
        return dict(self.launch_index.get(tag, {}).get("suggestion") or {})

    def _config_key(self, tag):
        ov = self._overrides_for_tag(tag)
        return json.dumps(ov, sort_keys=True, default=str)

    def _already_confirmed_config(self, tag):
        """CARBS re-samples its incumbent to estimate noise; the runs are deterministic
        per seed, so confirming the same overrides twice repeats the same numbers."""
        key = self._config_key(tag)
        # only real confirmations (not tags that were themselves skipped as duplicates)
        # and pending tags count, otherwise the original looks like a duplicate after a resume
        real = {c["tag"] for c in self.confirmations
                if not str(c.get("reason", "")).startswith("duplicate_config_of")}
        others = [t for t in (real | self.pending_confirm_tags)
                  if t != tag and self._config_key(t) == key]
        if others:
            append_ledger(self.ledger_path, {"event": "confirmation", "tag": tag,
                                              "confirmed_win": None, "confirmed_strong": None,
                                              "reason": f"duplicate_config_of_{others[0]}"})
            self.confirmations.append({"tag": tag, "confirmed": False, "confirmed_win": None,
                                       "confirmed_strong": None, "per_seed": None, "means": None,
                                       "stds": None, "reason": f"duplicate_config_of_{others[0]}"})
            return True
        return False

    def _overrides_for_tag(self, tag):
        return dict(self.launch_index.get(tag, {}).get("overrides") or {})

    def _maybe_confirm(self, tag):
        if any(c.get("tag") == tag for c in self.confirmations):
            return
        seeds = [self.task_spec["search_seed"]] + list(self.task_spec["confirm_seeds"])
        if any((tag, s) in self.gave_up for s in seeds):
            record = {"tag": tag, "confirmed": False, "confirmed_win": False,
                       "confirmed_strong": False, "per_seed": None, "means": None, "stds": None,
                       "reason": "seed_gave_up"}
            self.confirmations.append(record)
            append_ledger(self.ledger_path, {"event": "confirmation", "tag": tag,
                                              "confirmed_win": False, "confirmed_strong": False,
                                              "reason": "seed_gave_up"})
            self.pending_confirm_tags.discard(tag)
            return
        results = []
        for s in seeds:
            p = self._seed_json_path(tag, s)
            if not os.path.exists(p):
                return
            results.append(read_json(p))
        agg = aggregate_confirmation(results, self.task_spec, self.task_spec["win"],
                                      self.task_spec.get("strong_win", self.task_spec["win"]))
        overrides = self._overrides_for_tag(tag)
        if self.task_spec.get("observe_confirmation_mean"):
            from carbs import ObservationInParam
            mean_out, mean_cost = confirmation_mean_observation(results, self.task_spec)
            # CARBS groups observations of one input and compares the group's MEAN cost
            # with a quantile of the individual costs: a confirmation mean charged the
            # unverified seeds' wall time (1163 s next to the trial's 50 s) left no group
            # below the quantile and crashed every suggest/observe (robot arm, 2026-10-10).
            # The mean output is the information; the cost stays the search trial's own.
            search_res = [r for r in results if r.get("seed") == self.task_spec.get("search_seed", 1)]
            if search_res:
                mean_cost = observation_for_result(search_res[0], self.task_spec)[1]
            suggestion = self._find_suggestion_for_tag(tag)
            suggestion.pop(SUGGESTION_ID_KEY, None)
            try:
                self.carbs.observe(ObservationInParam(input=suggestion, output=mean_out, cost=mean_cost))
                self._save_carbs()
                append_ledger(self.ledger_path, {"event": "observed", "tag": tag, "output": mean_out,
                                                  "cost": mean_cost, "is_failure": False,
                                                  "kind": "confirmation_mean"})
            except ValueError as exc:
                # CARBS 2026-10-10: get_pareto_groups_conservative raised "max() arg is an
                # empty sequence" on the robot arm confirmation mean; the driver crashed on
                # every resume. The confirmation itself is recorded below regardless.
                append_ledger(self.ledger_path, {"event": "carbs_observe_failed", "tag": tag,
                                                  "output": mean_out, "cost": mean_cost,
                                                  "kind": "confirmation_mean", "error": str(exc)})
        record = {"tag": tag, **agg}
        self.confirmations.append(record)
        append_ledger(self.ledger_path, {"event": "confirmation", "tag": tag,
                                          "per_seed": agg["per_seed"], "means": agg["means"],
                                          "stds": agg["stds"], "all_verified": agg["all_verified"],
                                          "confirmed_win": agg["confirmed_win"],
                                          "confirmed_strong": agg["confirmed_strong"]})
        self.pending_confirm_tags.discard(tag)
        if agg["confirmed_win"]:
            self._write_win(tag, seeds, results, agg, overrides)
        self._write_status()
        stop_on = self.task_spec.get("stop_on", "none")
        satisfied = (agg["confirmed_strong"] if stop_on == "strong_win"
                     else agg["confirmed_win"] if stop_on == "win" else False)
        if satisfied:
            self._write_done("win", tag)

    def _write_win(self, tag, seeds, results, agg, overrides):
        import shutil
        win_dir = os.path.join(self.wins_dir, tag)
        os.makedirs(win_dir, exist_ok=True)
        for seed, result in zip(seeds, results):
            ckpt = result.get("model_path")
            if ckpt and os.path.exists(ckpt):
                shutil.copy2(ckpt, os.path.join(win_dir, f"seed{seed}.pth"))
        write_json_atomic(os.path.join(win_dir, "result.json"), {
            "task": self.task, "method": self.method, "tag": tag, "seeds": seeds,
            "overrides": overrides, **agg,
        })

    def _write_done(self, status, tag):
        write_json_atomic(self.done_path, {"status": status, "tag": tag})
        append_ledger(self.ledger_path, {"event": "done", "status": status, "tag": tag})

    # ---- retry / crash / timeout ----

    def _handle_crash(self, ev, reason):
        tag, seed = ev["tag"], ev["seed"]
        attempt = ev.get("attempt", 1)
        append_ledger(self.ledger_path, {"event": "finished", "tag": tag, "seed": seed,
                                          "outcome": reason})
        if attempt <= self.task_spec.get("max_retries", 0):
            kind = ev.get("kind", "search")
            overrides = ev.get("overrides", {})
            suggestion = ev.get("suggestion")
            if not self._launch(tag, seed, kind, attempt + 1, overrides, suggestion):
                self.pending_retries.append({"tag": tag, "seed": seed, "kind": kind,
                                              "attempt": attempt + 1, "overrides": overrides,
                                              "suggestion": suggestion})
            return
        self.gave_up.add((tag, seed))
        if ev.get("kind") == "confirm":
            self.pending_confirm_tags.discard(tag)
            return
        if ev.get("kind") == "search":
            from carbs import ObservationInParam
            suggestion = ev.get("suggestion") or self._find_suggestion_for_tag(tag)
            self.carbs.observe(ObservationInParam(input=dict(suggestion), output=0.0, cost=1.0,
                                                    is_failure=True))
            self._save_carbs()
            self.search_observed += 1
            append_ledger(self.ledger_path, {"event": "observed", "tag": tag, "output": 0.0,
                                              "cost": 1.0, "is_failure": True})

    def _handle_timeout(self, trial):
        try:
            os.killpg(trial.pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.time() + KILL_GRACE_S
        while time.time() < deadline and os.path.exists(f"/proc/{trial.pid}"):
            time.sleep(1.0)
        if os.path.exists(f"/proc/{trial.pid}"):
            try:
                os.killpg(trial.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if trial.slot_fd is not None:
            release_slot(trial.slot_fd)
        append_ledger(self.ledger_path, {"event": "finished", "tag": trial.tag, "seed": trial.seed,
                                          "outcome": "timeout"})
        if trial.kind == "search":
            from carbs import ObservationInParam
            cost = max(float(self.task_spec["trial_timeout_s"]), 1.0)
            self.carbs.observe(ObservationInParam(input=dict(trial.suggestion or {}), output=0.0,
                                                    cost=cost))
            self._save_carbs()
            self.search_observed += 1
            append_ledger(self.ledger_path, {"event": "observed", "tag": trial.tag, "output": 0.0,
                                              "cost": cost, "is_failure": False, "timeout": True})
        elif trial.kind == "confirm":
            self.gave_up.add((trial.tag, trial.seed))
            self.pending_confirm_tags.discard(trial.tag)
            if not any(c.get("tag") == trial.tag for c in self.confirmations):
                record = {"tag": trial.tag, "confirmed": False, "confirmed_win": False,
                           "confirmed_strong": False, "per_seed": None, "means": None,
                           "stds": None, "reason": "confirm_timeout"}
                self.confirmations.append(record)
                append_ledger(self.ledger_path, {"event": "confirmation", "tag": trial.tag,
                                                  "confirmed_win": False, "confirmed_strong": False,
                                                  "reason": "confirm_timeout"})

    def _poll_once(self):
        now = time.time()
        any_finished = False
        for key, trial in list(self.inflight.items()):
            timeout_s = self.task_spec["trial_timeout_s"]
            if now - trial.start_time > timeout_s:
                del self.inflight[key]
                self._handle_timeout(trial)
                any_finished = True
                continue
            alive = (trial.proc.poll() is None) if trial.proc is not None else \
                os.path.exists(f"/proc/{trial.pid}")
            if alive:
                continue
            del self.inflight[key]
            if trial.slot_fd is not None:
                release_slot(trial.slot_fd)
            json_path = self._seed_json_path(trial.tag, trial.seed)
            log_text = ""
            if os.path.exists(trial.log_path):
                with open(trial.log_path, "rb") as f:
                    log_text = f.read().decode(errors="replace")
            outcome, detail = classify_finish(os.path.exists(json_path), log_text,
                                               self.global_spec["license_error_regex"])
            if outcome == "ok":
                result = read_json(json_path)
                if trial.kind == "search":
                    self._observe_search_result(trial.tag, result)
                append_ledger(self.ledger_path, {"event": "finished", "tag": trial.tag,
                                                  "seed": trial.seed, "outcome": "ok"})
                self._maybe_confirm(trial.tag)
            else:
                self.last_error = f"{trial.tag}/seed{trial.seed}: {outcome} ({detail})"
                self._handle_crash({"tag": trial.tag, "seed": trial.seed, "attempt": trial.attempt,
                                     "kind": trial.kind, "overrides": trial.overrides,
                                     "suggestion": trial.suggestion}, outcome)
            any_finished = True
        if any_finished:
            self._write_status()

    # ---- status ----

    def _write_status(self):
        n_search_running = sum(1 for t in self.inflight.values() if t.kind == "search")
        n_confirm_running = sum(1 for t in self.inflight.values() if t.kind == "confirm")
        status = {
            "task": self.task, "method": self.method, "spec": self.spec_name,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "search_trials_launched": self.search_index,
            "search_trials_observed": self.search_observed,
            "verified_count": self.verified_count,
            "feasible_count": self.feasible_count,
            "best_feasible": self.best_feasible,
            "n_confirmations": len(self.confirmations),
            "confirmations": self.confirmations,
            "running": {"search": n_search_running, "confirm": n_confirm_running,
                        "tags": sorted({t.tag for t in self.inflight.values()})},
            "warm_start": self.warm_start_info,
            "last_error": self.last_error,
            "max_trials": self.task_spec["max_trials"],
        }
        write_json_atomic(self.status_path, status)
        self._write_global_status_txt()

    def _write_global_status_txt(self):
        state_root = os.path.join(self.root, "state")
        rows = []
        for name in sorted(os.listdir(state_root)) if os.path.isdir(state_root) else []:
            sp = os.path.join(state_root, name, "status.json")
            if not os.path.exists(sp):
                continue
            try:
                s = read_json(sp)
            except (OSError, json.JSONDecodeError):
                continue
            done_p = os.path.join(state_root, name, "DONE")
            done = read_json(done_p) if os.path.exists(done_p) else None
            rows.append((name, s, done))
        lines = [f"Campaign status: {datetime.now().isoformat(timespec='seconds')}", ""]
        header = (f"{'spec':<24}{'launched':>9}{'observed':>9}{'verified':>9}{'feasible':>9}"
                  f"{'best_tfr%':>10}{'cost_s':>9}{'calls':>6}{'confirms':>9}  status")
        lines.append(header)
        lines.append("-" * len(header))
        for name, s, done in rows:
            best = s.get("best_feasible") or {}
            status_str = f"DONE({done['status']}/{done.get('tag')})" if done else "running"
            lines.append(
                f"{name:<24}{s.get('search_trials_launched', 0):>9}"
                f"{s.get('search_trials_observed', 0):>9}{s.get('verified_count', 0):>9}"
                f"{s.get('feasible_count', 0):>9}"
                f"{best.get('tfr_pct', float('nan')):>10.2f}{best.get('cost', float('nan')):>9.0f}"
                f"{str(best.get('calls', '')):>6}"
                f"{s.get('n_confirmations', 0):>9}  {status_str}"
            )
        tmp = os.path.join(self.root, f"status.txt.tmp{os.getpid()}")
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, os.path.join(self.root, "status.txt"))

    # ---- main loop ----

    def run(self, check_only=False):
        self.verify()
        fresh = not os.path.exists(self.carbs_pkl)
        if fresh:
            self._build_fresh_carbs()
            self.do_warm_start()
            self._save_carbs()
            self._write_status()
            if check_only:
                return
            self._main_loop()
            return
        self._load_carbs()
        if check_only:
            return
        self.resume()
        self._write_status()
        self._main_loop()

    def _fill_slots(self):
        parallel = self.task_spec["parallel"]
        max_trials = self.task_spec["max_trials"]
        still_pending = []
        for r in self.pending_retries:
            if len(self.inflight) >= parallel:
                still_pending.append(r)
                continue
            if (r["tag"], r["seed"]) in self.inflight or os.path.exists(
                    self._seed_json_path(r["tag"], r["seed"])):
                continue
            if not self._launch(r["tag"], r["seed"], r["kind"], r["attempt"],
                                 r["overrides"], r["suggestion"]):
                still_pending.append(r)
        self.pending_retries = still_pending
        for tag in list(self.pending_confirm_tags):
            if len(self.inflight) >= parallel:
                return
            self._launch_confirm(tag, self._overrides_for_tag(tag))
        while len(self.inflight) < parallel and self.search_index < max_trials:
            if not self._launch_search():
                break

    def _main_loop(self):
        while True:
            if os.path.exists(self.done_path):
                break
            self._fill_slots()
            self._poll_once()
            if os.path.exists(self.done_path):
                break
            if (self.search_observed >= self.task_spec["max_trials"]
                    and not self.inflight and not self.pending_confirm_tags
                    and not self.pending_retries):
                self._write_done("exhausted", None)
                break
            time.sleep(self.poll_interval)
        while self.inflight or self.pending_retries:
            self._fill_slots()
            self._poll_once()
            if self.inflight or self.pending_retries:
                time.sleep(self.poll_interval)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--task", required=True, help="spec name: specs/<name>.json")
    ap.add_argument("--check", action="store_true",
                    help="validate the spec and (if fresh) run warm start, without launching trials")
    args = ap.parse_args()

    driver = Driver(args.root, args.task)
    driver.run(check_only=args.check)
    if args.check:
        print(f"OK: spec {args.task!r} valid for task {driver.task!r} / method {driver.method!r} "
              f"(warm_start={driver.warm_start_info})")


if __name__ == "__main__":
    main()
