import os
import json

root_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(root_dir, "par_study_weight_scale")
config = {}

weight_scale = [1, 5, 10, 25, 50, 100, 500]

os.mkdir(target_dir)
for scale in weight_scale:
    config["dims"] = [9, 30, 3]
    config["weight_scale"] = scale
    config["network_type"] = "snest"
    config_name = f"bars_le_weight_scale_{int(scale*100)}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config, f, indent=4)
