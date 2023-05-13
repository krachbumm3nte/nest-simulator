import os
import json
from copy import deepcopy


root_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(root_dir, "par_study_function_approximator")


dims_teacher = [15, 10, 5]

config = {
    "t_pres": 50,
    "out_lag": 35,
    "init_self_pred": True,
    "mode": "teacher",
    "dims_teacher": deepcopy(dims_teacher),
    "dims": deepcopy(dims_teacher),
    "test_interval": 500,
    "n_epochs": 5000,
    "noise": True,
    "sigma": 0.1,
    "teacher_weights": "init_weights_15_10_5.json",
    "eta": {
        "ip": [
            0.0001,
            0.0
        ],
        "pi": [
            0.0,
            0.0
        ],
        "up": [
            0.00025,
            0.00005
        ],
        "down": [
            0,
            0.0
        ]
    }


}

n_hidden = [2, 5, 10, 15]


os.mkdir(target_dir)
for n in n_hidden:
    config["dims"][1] = n

    config_name = f"teacher_n_hidden_{n}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config, f, indent=4)
