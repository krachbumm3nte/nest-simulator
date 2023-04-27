import os
import json
import numpy as np

root_dir = os.path.dirname(os.path.realpath(__file__))

base_file = os.path.join(root_dir, "self_prediction/self_pred_snest.json")

target_dir = os.path.join(root_dir, "par_study_dropout")
p_conn = np.arange(0.1, 1, 0.2)

with open(base_file, "r") as f:
    config = json.load(f)


os.mkdir(target_dir)
for p in p_conn:
    config["p_conn"] = p
    config_name = f"selfpred_dropout_p_{int(10*p)}.json"
    with open(os.path.join(target_dir, config_name), "w") as f:
        json.dump(config, f, indent=4)
