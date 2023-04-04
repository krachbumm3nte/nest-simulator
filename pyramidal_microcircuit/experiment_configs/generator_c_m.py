import os
import json

root_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(root_dir, "c_m")

weight_scale_list = [1, 5, 10]
c_m_list = [1, 2, 5, 10]


config = {
    "sim_time": 100,
    "out_lag": 60.0,
    "latent_equilibrium": True,
    "dims": [9, 30, 3],
    "eta": {
        "ip": [
            0.004,
            0.0
        ],
        "pi": [
            0.0,
            0.0
        ],
        "up": [
            0.01,
            0.002
        ],
        "down": [
            0.0,
            0.0
        ]
    },
    "record_interval": 0.5,
    "dims": [
        9,
        30,
        3
    ],
    "weight_scale": 0.25,
    "network_type": "snest"
}


os.mkdir(target_dir)
for scale in weight_scale_list:
    for c_m in c_m_list:
        config["weight_scale"] = scale
        config["network_type"] = "snest"
        config["c_m_api"] = c_m
        config_name = f"bars_le_weight_scale_{int(scale*100)}_c_m_{c_m}.json"
        with open(os.path.join(target_dir, config_name), "w") as f:
            json.dump(config, f, indent=4)
