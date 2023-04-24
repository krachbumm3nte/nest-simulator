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
    "snest": {
        "type": NestNetwork,
        "spiking": True,
        "name": "NEST spiking"
    },
    "rnest": {
        "type": NestNetwork,
        "spiking": False,
        "name": "NEST rate"
    },
    "numpy": {
        "type": NumpyNetwork,
        "spiking": False,
        "name": "NumPy"
    },
}


results = {}

all_dims = [[9, 10, 3], [9, 30, 3], [9, 200, 3]]

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
        params.train_samples = 1

        utils.setup_nest(params)
        net = conf["type"](params)
        net.set_all_weights(wgts)

        t_start = time()
        net.train_epoch()
        t_stop = time()
        train_time = t_stop - t_start
        results[n_hidden][conf["name"]] = train_time
        nest.ResetKernel()
        print("Done.\n")

with open(os.path.join(plot_dir, "data.json"), "w") as f:
    json.dump(results, f)


fig, ax = plt.subplots(1, len(all_dims))

for i, (n_hidden, data) in enumerate(results):
    ax[i].bar(data.keys(), data.values(), width = 0.55)
    ax[i].set_title(r"$n_{{hidden}} = {}$".format(n_hidden))

ax[0].set_ylabel(r"$T_{{sim}} [s]$")
plt.savefig(os.path.join(plot_dir, "plot.png"))
