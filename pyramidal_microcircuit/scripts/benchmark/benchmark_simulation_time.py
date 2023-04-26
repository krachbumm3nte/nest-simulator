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
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_hidden", "--list", nargs="+",
                        help="list of hidden neurons with which to run the benchmark")
    parser.add_argument("--target_dir",
                        type=str,
                        help="directory in which to store")
    parser.add_argument("--threads",
                        type=int,
                        default=8,
                        help="number of threads to allocate. Only has an effect when simulating with NEST.")
    parser.add_argument("--n_samples",
                        type=int,
                        default=10,
                        help="number samples for each configuration")
    parser.add_argument("--t_pres",
                        type=int,
                        default=100,
                        help="Presentation time per sample in ms.")
    args = parser.parse_args()
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
            "weight_scale": 1

        },
        "rnest": {
            "type": NestNetwork,
            "spiking": False,
            "name": "NEST rate",
            "weight_scale": 1
        },
        # "numpy": {
        #     "type": NumpyNetwork,
        #     "spiking": False,
        #     "name": "NumPy",
        #     "weight_scale": 1
        # },
    }

    results = {}

    for i, n_hidden in enumerate(args.n_hidden):
        n_hidden = int(n_hidden)
        dims = [9, n_hidden, 3]
        wgts = utils.generate_weights(dims)

        results[n_hidden] = {}
        for conf in configs.values():
            print(f"simulating {conf['name']} on {dims}")
            params = Params()
            params.spiking = conf["spiking"]
            params.dims = dims
            params.out_lag = 0
            params.t_pres = args.t_pres
            params.weight_scale = conf["weight_scale"]
            params.threads = args.threads

            utils.setup_nest(params)
            net = conf["type"](params)
            net.set_all_weights(wgts)
            net.train_samples = args.n_samples

            t_start = time()
            net.train_epoch()
            t_stop = time()
            train_time = t_stop - t_start
            results[n_hidden][conf["name"]] = train_time
            nest.ResetKernel()
            print("Done.\n")

    data = args.__dict__
    data["times"] = results

    with open(args.target_dir, "w") as f:
        json.dump(data, f)
