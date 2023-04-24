import json
import os
import sys
import nest

import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params
import src.plot_utils as plot_utils

import nest
from time import time
import matplotlib.pyplot as plt

args = sys.argv[1:]

n_threads = args[0]

plot_dir = args[1]

plot_utils.setup_plt()

configs = {
    "snest_high": {
        "type": NestNetwork,
        "spiking": True,
        "name": "NEST spiking",
        "weight_scale": 250

    },
    "snest_low": {
        "type": NestNetwork,
        "spiking": True,
        "name": "NEST spiking",
        "weight_scale": 10

    },
    "rnest": {
        "type": NestNetwork,
        "spiking": False,
        "name": "NEST rate",
        "weight_scale": 10
    },
    "numpy": {
        "type": NumpyNetwork,
        "spiking": False,
        "name": "NumPy",
        "weight_scale": 10
    },
}


results = {}

all_dims = [[9, 30, 3], [9, 100, 3], [9, 200, 3]]

for i, dims in enumerate(all_dims):
    wgts = utils.generate_weights(dims)
    n_hidden = dims[1]

    results[n_hidden] = {}
    for conf in configs.values():
        print(f"simulating {conf['name']} on {dims}")
        params = Params()
        params.spiking = conf["spiking"]
        params.dims = dims
        params.out_lag = 50
        params.sim_time = 100
        params.weight_scale = conf["weight_scale"]

        utils.setup_nest(params)
        net = conf["type"](params)
        net.set_all_weights(wgts)
        net.train_samples = 50

        t_start = time()
        net.train_epoch()
        t_stop = time()
        train_time = t_stop - t_start
        results[n_hidden][conf["name"]] = train_time
        nest.ResetKernel()
        print("Done.\n")

with open(os.path.join(plot_dir, "data.json"), "w") as f:
    json.dump(results, f)
