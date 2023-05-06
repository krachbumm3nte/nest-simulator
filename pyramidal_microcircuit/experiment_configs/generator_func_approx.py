import os
import json

root_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(root_dir, "par_study_function_approximator")
config = {}

dims_teacher = [30, 20, 10]
n_hidden = [10, 15, 20, 25, 30, 35, 40]


os.mkdir(target_dir)
for n in n_hidden:
    config["t_pres"] = 100
    config["out_lag"] = 60
    config["init_self_pred"] = False
    config["mode"] = "teacher"
    config["dims_teacher"] = dims_teacher
    config["dims"] = dims_teacher
    config["dims"][1] = n
    config["eta"] = {
        "ip": [
            0.002,
            0.0
        ],
        "pi": [
            0.005,
            0.0
        ],
        "up": [
            0.005,
            0.0015
        ],
        "down": [
            0,
            0.0
        ]
    }
    config_name = f"teacher_n_hidden_{n}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config, f, indent=4)
