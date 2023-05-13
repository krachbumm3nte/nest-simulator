import os
import json
from copy import deepcopy

root_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(root_dir, "par_study_c_m_psi")

psi_list = [10, 50, 100]
c_m_list = [1, 10, 50]


default_config = {
    "t_pres": 200,
    "out_lag": 130.0,
    "latent_equilibrium": True,
    "dims": [9, 30, 3],
    "eta": {
        "ip": [
            0.001,
            0.0
        ],
        "pi": [
            0.0025,
            0.0
        ],
        "up": [
            0.0025,
            0.0005
        ],
        "down": [
            0.0,
            0.0
        ]
    },
    "network_type": "snest",
    "psi": 100,
    "n_epochs": 500,
    "C_m_api": 1
}


os.mkdir(target_dir)
for psi in psi_list:
    for c_m in c_m_list:
        config = deepcopy(default_config)
        config["psi"] = psi
        config["C_m_api"] = c_m
        config_name = f"bars_le_psi_{psi}_c_m_{c_m}.json"
        with open(os.path.join(target_dir, config_name), "w") as f:
            json.dump(config, f, indent=4)
