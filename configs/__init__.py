"""Per-benchmark experiment configurations.

Each task module defines TASK (system, paths, seeds) and one dict per method
(PGD_MILP, ENUM_MILP) holding the keyword arguments of the training loop.
``get_config`` merges them with optional overrides:

    cfg = get_config("pendulum", "enum_milp", {"lr": 3e-4})
"""

import copy
import importlib

TASKS = [
    "pendulum",
    "unicycle",
    "cartpole",
    "robot_arm",
    "roboracer",
]

# method name -> config attribute. "enum_lean" is an alias of the one canonical
# enumeration implementation (kept for result tags and dispatch only).
METHODS = {
    "pgd_milp": "PGD_MILP",
    "enum_milp": "ENUM_MILP",
    "enum_lean": "ENUM_MILP",
}

# Consumed by the runner, not by the training loops.
RUNNER_KEYS = {"f_pi_source", "h_source", "verifier"}


def get_config(task: str, method: str, overrides: dict = None) -> dict:
    """Return {"task": ..., "method_kwargs": ...} for a task/method pair."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {list(METHODS)}")

    module = importlib.import_module(f"configs.{task}")
    attr = METHODS[method]
    if not hasattr(module, attr):
        raise ValueError(f"task {task!r} does not define method {method!r}")

    task_info = copy.deepcopy(module.TASK)
    method_kwargs = copy.deepcopy(getattr(module, attr))

    if overrides:
        known = set(method_kwargs) | RUNNER_KEYS
        # milp__* / milp_fast__* configure the verifier itself (examples/run_finetune.py)
        unknown = {k for k in set(overrides) - known
                   if not k.startswith(("milp__", "milp_fast__"))}
        if unknown:
            raise KeyError(f"unknown override keys: {sorted(unknown)}")
        method_kwargs.update(overrides)

    return {"task": task_info, "method_kwargs": method_kwargs}
