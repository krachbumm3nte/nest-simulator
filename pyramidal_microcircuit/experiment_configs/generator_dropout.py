import os
import json
import numpy as np

root_dir = os.path.dirname(os.path.realpath(__file__))

target_dir = os.path.join(root_dir, "par_study_dropout")
p_conn = np.round(np.arange(0.6, 1.01, 0.05), 2)

config = {
    "t_pres": 50,
    "out_lag": 0,
    "latent_equilibrium": True,
    "psi": 100,
    "mode": "self-pred",
    "init_self_pred": False,
    "test_interval": -1,
    "dims": [
        8,
        8,
        8
    ],
    "n_epochs": 2000,
    "store_errors": True,
    "eta": {
        "ip": [
            0.02375,
            0.0
        ],
        "pi": [
            0.05,
            0.0
        ],
        "up": [
            0,
            0
        ],
        "down": [
            0.0,
            0.0
        ]
    },
    "network_type": "snest"
}

os.mkdir(target_dir)
for p in p_conn:
    print(p)
    config["p_conn"] = p
    config_name = f"selfpred_dropout_p_{int(np.round(100*p))}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config, f, indent=4)
