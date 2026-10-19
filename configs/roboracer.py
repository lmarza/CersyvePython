"""RoboRacer tube benchmark (RoboRacer): the certificate that drove the car.

The system and models/roboracer/V_pretrain.pth are frozen; only the loop
parameters may be tuned. The deployed certificate: models/roboracer/car_validated/.
"""
from ._base import PGD_MILP_BASE, ENUM_MILP_BASE

TASK = {
    "system": "systems.roboracer:RoboRacer",
    "data_path": None,            # dynamics and constraint are analytical
    "model_dir": "models/roboracer",
    "f_pi_source": "analytical",
    "h_source": "analytical",
    "seeds": [1, 0, 2],
}

MILP_TIMEOUT = 300
MAX_ITER = 50000

# PGD+MILP: base settings with max_skip 150.
PGD_MILP = {**PGD_MILP_BASE, "max_skip": 150, "max_iter": MAX_ITER,
            "milp_timeout": MILP_TIMEOUT}

ENUM_MILP = {**ENUM_MILP_BASE, "min_region_frac": 0.1, "max_skip": 200, "max_iter": MAX_ITER,
             "milp_timeout": MILP_TIMEOUT}
