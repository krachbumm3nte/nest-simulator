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
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        help="configuration file")
    parser.add_argument("--out_file",
                        type=str,
                        help="file in which to store simulation results")
    args = parser.parse_args()
    plot_utils.setup_plt()

    with open(args.config) as f:
        config = json.load(f)

    config["results"] = {}
    n_reps = config["n_repetitions"] if "n_repetitions" in config else 1
    n_samples = config["n_samples"] if "n_samples" in config else 1

    for i, (name, param_dict) in enumerate(config["configs"].items()):
        print(f"Benchmarking {name}...")

        sim_times = []

        for n in range(n_reps):
            print(f"\trun {n+1}/{n_reps}...")

            params = Params()
            params.from_dict(config["global_params"])
            params.from_dict(param_dict)
            params.out_lag = 0 # ensures that simulation works for any t_pres

            utils.setup_nest(params)
            if params.network_type == "numpy":
                net = NumpyNetwork(params)
            elif params.network_type in ["rnest", "snest"]:
                net = NestNetwork(params)
            else:
                raise ValueError(f"invalid network type: {params.network_type}")

            if "init_weights" in config:
                with open(config["init_weights"]) as f:
                    wgts = json.load(f)
                net.set_all_weights(wgts)

            net.train_samples = n_samples

            t_start = time()
            net.train_epoch()
            t_stop = time()
            train_time = t_stop - t_start
            sim_times.append(train_time)
        t_mean = np.mean(sim_times)
        t_std = np.std(sim_times)
        config["results"][name] = {"times": sim_times,
                                   "t_mean": t_mean,
                                   "std": t_std}
        print(f"mean time: {t_mean}s, std:{t_std}s. \n\n")
        nest.ResetKernel()

    with open(args.out_file, "w") as f:
        json.dump(config, f)
