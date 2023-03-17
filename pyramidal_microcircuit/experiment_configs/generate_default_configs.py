import os
import json


t_pres = [0.3, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]  # presentation times in ms

eta_default = {
    "ip": [0.2, 0],
    "pi": [0, 0],
    "up": [0.5, 0.1],
    "down": [0, 0]
}


for le in [False, True]:
    for t in t_pres:
        # scales all learning rates. longer t_pres->smaller eta
        eta = {k: [lr/t for lr in v] for k, v in eta_default.items()}

        config = {
            "sim_time": t,
            "out_lag": round(0.6*t, 1),
            "tau_delta": 1,
            "tau_m": 1,
            "latent_equilibrium": le,
            "eta": eta
        }

        config_name = f"bars_{'le' if le else 'orig'}_tpres_{int(10*t)}.json"
        with open(os.path.join("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/experiment_configs", config_name), "w") as f:
            json.dump(config, f, indent=4)
