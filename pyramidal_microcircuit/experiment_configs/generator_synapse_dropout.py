import os
import json
import numpy as np

root_dir = os.path.dirname(os.path.realpath(__file__))
base_file = os.path.join(root_dir, "bars_le/bars_le_tpres_500.json")
target_dir = os.path.join(root_dir, "dropout")
with open(base_file, "r") as f:
    config = json.load(f)

p_conn = np.linspace(0.6, 1.0, 5, endpoint=True)

os.mkdir(target_dir)
for p in p_conn:
    config["p_conn"] = p
    config_name = f"dropout_p_{int(p*10)}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config, f, indent=4)
