from ._base import PGD_MILP_BASE, ENUM_MILP_BASE

TASK = {
    "system": "systems.robot_arm:RobotArm",
    "data_path": "data/robot_arm_data.npz",
    "model_dir": "models/robot_arm",
    "f_pi_source": "analytical",
    "h_source": "learned",
    "seeds": [1, 0, 2],
}

# Shared by both methods; a limit can only turn inconclusive verdicts into proven ones.
MILP_TIMEOUT = 1800

# PGD+MILP configuration of the table (results/pgd_milp).
PGD_MILP = {**PGD_MILP_BASE,
            "lr": 0.00027833200647262165, "search_size": 2847,
            "pgd_step": 20, "pgd_eps": 0.08516878407248273,
            "max_skip": 466, "reg_coef": 0.06831362732293038,
            "milp_timeout": MILP_TIMEOUT, "milp_max_inconclusive": 1}

# Enum+MILP defaults.
ENUM_MILP = {**ENUM_MILP_BASE,
             "lr": 7e-5, "capacity": 8000, "sample_size": 512, "replay": 3,
             "max_skip": 600,
             "enum_coverage": 80.0, "enum_n_trees": 300, "enum_n_train": 20000,
             "refill_threshold": 400,
             "fallback_eval_samples": 30000, "fallback_n_neighbors": 3000,
             "search_size": 1500, "eps_h": 0.015, "eps_v": 0.02, "reg_coef": 0.10,
             "milp_timeout": MILP_TIMEOUT}
