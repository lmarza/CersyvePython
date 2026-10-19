from ._base import PGD_MILP_BASE, ENUM_MILP_BASE

TASK = {
    "system": "systems.unicycle:Unicycle",
    "data_path": "data/unicycle_data.npz",
    "model_dir": "models/unicycle",
    "f_pi_source": "learned",
    "h_source": "learned",
    "seeds": [1, 0, 2],
}

# A proof on this task can need ~350 s; a time limit can only turn an inconclusive
# verdict into a proven one, never change a proven verdict. Shared by both methods.
MILP_TIMEOUT = 600

# PGD+MILP configuration of the table (results/pgd_milp_tuned_final).
PGD_MILP = {**PGD_MILP_BASE,
            "lr": 7.755661996296333e-05, "search_size": 1299,
            "pgd_step": 5, "pgd_eps": 0.05576690142072709,
            "max_skip": 784, "reg_coef": 0.22723712562446818,
            "milp_timeout": MILP_TIMEOUT}

# Enum+MILP defaults.
ENUM_MILP = {**ENUM_MILP_BASE, "max_skip": 2000, "milp_timeout": MILP_TIMEOUT}
