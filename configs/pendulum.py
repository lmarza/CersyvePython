from ._base import PGD_MILP_BASE, ENUM_MILP_BASE

TASK = {
    "system": "systems.pendulum:Pendulum",
    "data_path": "data/pendulum_data.npz",
    "model_dir": "models/pendulum",
    "f_pi_source": "learned",
    "h_source": "analytical",
    "seeds": [1, 0, 2],
}

MILP_TIMEOUT = 300

# PGD+MILP configuration of the table (results/pgd_milp_tuned_final).
PGD_MILP = {**PGD_MILP_BASE,
            "lr": 2.4385715665649566e-05, "search_size": 1183,
            "pgd_step": 3, "pgd_eps": 0.4563278994001354,
            "max_skip": 150, "reg_coef": 0.37708797963808083,
            "milp_timeout": MILP_TIMEOUT}

# Enum+MILP defaults.
ENUM_MILP = {**ENUM_MILP_BASE, "max_skip": 200, "milp_timeout": MILP_TIMEOUT}
