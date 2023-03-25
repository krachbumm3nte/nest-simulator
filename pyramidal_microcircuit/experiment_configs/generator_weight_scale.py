import os
import json

root_dir = os.path.dirname(os.path.realpath(__file__))
base_file = os.path.join(root_dir, "bars_le/bars_le_tpres_500.json")
target_dir = os.path.join(root_dir, "weight_scale")
with open(base_file, "r") as f:
    config = json.load(f)

weight_scale = [0.1, 0.25, 0.5, 1, 5, 10, 50, 100, 500, 1000]

os.mkdir(target_dir)
for scale in weight_scale:
    config["dims"] = [9, scale, 3]
    config_name = f"bars_le_weight_scale_{int(scale*100)}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config, f, indent=4)
