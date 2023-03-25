import os
import json
import sys

root_dir = os.path.dirname(os.path.realpath(__file__))

base_file = os.path.join(root_dir, "bars_le/bars_le_tpres_500.json")

target_dir = os.path.join(root_dir, "n_hidden")
n_hidden = [5, 10, 20, 30, 40, 50, 75, 100, 200]

with open(base_file, "r") as f:
    config = json.load(f)


os.mkdir(target_dir)
for n in n_hidden:
    config["dims"] = [9, n, 3]
    config_name = f"bars_le_n_hidden_{n}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config, f, indent=4)
