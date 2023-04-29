import os
import json
import numpy as np
from copy import deepcopy
root_dir = os.path.dirname(os.path.realpath(__file__))

target_dir = os.path.join(root_dir, "par_study_reset_strategies")
foo = {"no_reset": {"reset": 0},
       "soft_reset": {"reset": 1},
       "soft_reset_c_m": {"reset": 1,
                          "C_m_api": 75},
       "hard_reset": {"reset": 2},
       }

config = {
    "t_pres": 50,
    "out_lag": 35.0,
    "latent_equilibrium": True,
    "eta": {
        "ip": [
            0.004,
            0.0
        ],
        "pi": [
            0.01,
            0.0
        ],
        "up": [
            0.01,
            0.003
        ],
        "down": [
            0,
            0.0
        ]
    },
    "record_interval": 1,
    "init_self_pred": False,
    "network_type": "snest"
}

os.mkdir(target_dir)
for k, v in foo.items():
    config_copy = deepcopy(config)
    for a, b in v.items():
        config_copy[a] = b
    config_name = f"{k}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config_copy, f, indent=4)
