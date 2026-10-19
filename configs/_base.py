"""Hyperparameters shared by all benchmarks.

Both dictionaries list exactly the keyword arguments of the two training loops
(cersyve/finetune.py). Task modules copy them and apply their deltas. Keys that change the certification condition
(tol, milp_timeout, the verifier itself) are set per task and shared by both
methods; they are never searched.
"""

# finetune_value_w_MILP (PGD / boundary-guided search + MILP)
PGD_MILP_BASE = {
    "lr": 1e-4,
    "max_iter": 100000,
    "search_size": 1000,
    "bnd_ratio": 0.1,
    "bnd_ratio_avg": 0.9,
    "min_bnd_ratio": 0.01,
    "max_bnd_ratio": 1.0,
    "bnd_eps": 0.1,
    "search_method": "BGB",
    "pgd_step": 10,
    "pgd_eps": 0.1,
    "backtrack_step": 20,
    "length_discount": 0.8,
    "direct_discount": 0.5,
    "tol": 1e-4,
    "capacity": 10000,
    "sample_size": 100,
    "search_stop": 1000,
    "replay": 1,
    "max_skip": 1000,
    "reg_method": "ESR",
    "esr_max_con": None,
    "eps_h": 0.01,
    "eps_v": 0.01,
    "reg_coef": 0.1,
    "esr_box_ratio": 0.7,      # ESR states are drawn in the 0.7 centre box
    "milp_timeout": 300,
    # verify the full domain as soon as the search runs dry, no cooldown, no growing box
    "milp_start_iter": 0,
    "milp_cooldown": 0,
    "milp_box_shrink": 1.0,
    "milp_box_grow": 1.0,
    "milp_box_max": 1.0,
    "milp_max_inconclusive": 6,
    "min_region_frac": 0.0,    # region floor: refuse to verify a (near) empty set (0 = off)
    "eval_every": 10,
    "save_every": 1000,
}

# finetune_value_enum_MILP (enumeration + MILP)
ENUM_MILP_BASE = {
    "lr": 1e-4,
    "max_iter": 100000,
    "tol": 1e-4,
    "capacity": 10000,
    "sample_size": 100,
    "replay": 1,
    "max_skip": 1000,
    "enum_freq": 1000,
    "enum_n_samples": 20000,
    "enum_coverage": 70.0,
    "enum_n_trees": 2000,
    "enum_max_depth": 20,
    "enum_n_train": 40000,
    "enum_n_jobs": None,       # None follows OMP_NUM_THREADS
    "refill_threshold": 500,
    "fallback_eval_rounds": 5,
    "fallback_eval_samples": 100000,
    "fallback_n_neighbors": 5000,
    "fallback_radius": 0.05,
    "search_size": 1000,
    "reg_method": "ESR",
    "esr_max_con": None,
    "eps_h": 0.01,
    "eps_v": 0.01,
    "reg_coef": 0.1,
    "esr_box_ratio": 1.0,      # ESR states are drawn on the full domain
    "milp_timeout": 300,
    "min_region_frac": 0.0,    # region floor: refuse to verify a (near) empty set (0 = off)
    "max_skip_halving": 1,     # 1 = halve max_skip after a refutation (floor 50); 0 = constant
    "eval_every": 100,
    "save_every": 5000,
}
