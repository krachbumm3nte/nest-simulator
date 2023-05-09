import os
import json
from copy import deepcopy


root_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(root_dir, "par_study_function_approximator")
config = {}

dims_teacher = [15, 10, 5]
n_hidden = [4, 6, 8, 10, 15]


os.mkdir(target_dir)
for n in n_hidden:
    config["t_pres"] = 50
    config["out_lag"] = 35
    config["init_self_pred"] = False
    config["mode"] = "teacher"
    config["dims_teacher"] = deepcopy(dims_teacher)
    config["dims"] = deepcopy(dims_teacher)
    config["dims"][1] = n
    config["test_interval"] = 500
    config["n_epochs"] = 5000
    config["teacher_weights"] = "init_weights_15_10_5.json"
    config["eta"] = {
        "ip": [
            0.0012,
            0.0
        ],
        "pi": [
            0.006,
            0.0
        ],
        "up": [
            0.0012,
            0.0005
        ],
        "down": [
            0,
            0.0
        ]
    }
    config_name = f"teacher_n_hidden_{n}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config, f, indent=4)
