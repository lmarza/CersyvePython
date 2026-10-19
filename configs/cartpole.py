from ._base import PGD_MILP_BASE, ENUM_MILP_BASE

TASK = {
    "system": "systems.cartpole:CartPole",
    "data_path": "data/cartpole_data.npz",
    "model_dir": "models/cartpole",
    "f_pi_source": "learned",
    "h_source": "analytical",
    "seeds": [1, 0, 2],
}

MILP_TIMEOUT = 500

# PGD+MILP configuration of the table (results/pgd_milp_tuned_final).
PGD_MILP = {**PGD_MILP_BASE,
            "lr": 0.0009776166093634973, "search_size": 1041,
            "pgd_step": 37, "pgd_eps": 0.082604054891454,
            "max_skip": 74, "reg_coef": 0.12289144377205136,
            "milp_timeout": MILP_TIMEOUT}

# Enum+MILP defaults.
ENUM_MILP = {**ENUM_MILP_BASE, "min_region_frac": 0.1, "max_skip": 200, "milp_timeout": MILP_TIMEOUT}
