import os
import json
import sys
from copy import deepcopy
root_dir = os.path.dirname(os.path.realpath(__file__))

base_file = os.path.join(root_dir, "bars_le_full_plast_deep.json")

target_dir = os.path.join(root_dir, "par_study_deep_plast")

foo = 0.01
eta = {
        "ip": [
            0.,
            0.,
            0.0
        ],
        "pi": [
            0.0,
            0.0,
            0.0
        ],
        "up": [
            0.0,
            0.0,
            0.0
        ],
        "down": [
            0,
            0,
            0
        ]
    }
with open(base_file, "r") as f:
    config = json.load(f)

config["sim_time"] = 25
config["out_lag"] = 12
config["n_epochs"] = 10

os.mkdir(target_dir)
for name in ["up", "ip", "pi"]:
    for i in range(3):
        if i > 1 and name != "up":
            continue    
        config["eta"] = deepcopy(eta)
        config["eta"][name][i] = foo
        config_name = f"deep_{name}_{i}.json"
        with open(os.path.join(target_dir, config_name), "w") as f:
            json.dump(config, f, indent=4)
