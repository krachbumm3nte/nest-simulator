import argparse
import os
import sys
import time
import json
from datetime import timedelta

import numpy as np
import src.utils as utils
from microcircuit_learning import run_simulations
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params

import nest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="parameter_study.py",
        description="Processes multiple configurations in sequence, with the option " +
                    "to unify some common simulation parameters.")
    parser.add_argument("--network",
                        type=str, choices=["numpy", "rnest", "snest"],
                        help="Type of network to train. Choice between exact mathematical simulation ('numpy') and" +
                        "NEST simulations with rate- or spiking neurons ('rnest', 'snest')")
    parser.add_argument("--config_dir",
                        type=str,
                        help="folder in which to search for config files.")
    parser.add_argument("--target_dir",
                        type=str,
                        help="directory in which to store results.")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="automatically overwrite existing simulations in the target directory.")
    parser.add_argument("--threads",
                        type=int,
                        default=8,
                        help="number of threads to allocate. Only has an effect when simulating with NEST.")
    parser.add_argument("--weights",
                        type=str,
                        help="Set of initial weights to be used for all simulations. Should be a path to a .json file.")

    args = parser.parse_args()

    all_configs = os.listdir(args.config_dir)

    init_weights = None
    if args.weights:
        print(f"initializing all simulations with weights from {args.weights}")
        with open(args.weights) as f:
            init_weights = json.load(f)

    global_t_start = time.time()
    for i, config in enumerate(all_configs):
        print(f"\nprocessing config file {config}...")
        if not config.endswith(".json"):
            print(f"skipping file {config}")
            continue

        params = Params(os.path.join(args.config_dir, config))
        if args.network:
            if params.network_type != args.network:
                print(f"WARNING: both input file and script parameters specify " +
                      f"different network types ({params.network_type}/{args.network}).")
            params.network_type = args.network
        elif not params.network_type:
            print("no network type specified, aborting.")
            sys.exit()

        print(f"Preparing simulation for network type: {params.network_type}")

        spiking = params.network_type == "snest"
        params.spiking = spiking
        params.threads = args.threads
        use_nest = params.network_type != "numpy"

        config_name = os.path.split(config)[-1].split(".")[0]
        root_dir, imgdir, datadir = utils.setup_directories(
            name=config_name, type=params.network_type, root=args.target_dir)
        if not root_dir:
            print("\ta simulation of that name already exists, skipping.\n")
            continue
        print(f"created dirs: {root_dir}")

        if not use_nest:
            net = NumpyNetwork(params)
            if init_weights:
                net.set_all_weights(init_weights)
        else:
            utils.setup_nest(params, datadir)
            net = NestNetwork(params, init_weights)

        simulation_times = []

        # dump simulation parameters and initial weights to .json files
        params.to_json(os.path.join(root_dir, "params.json"))
        init_weight_fp = os.path.join(root_dir, "init_weights.json")
        if init_weights:
            with open(init_weight_fp, "w") as f:
                json.dump(init_weights, f, indent=4)
        else:
            utils.store_synaptic_weights(net, init_weight_fp)

        print(f"Setup complete, beginning to train...")
        run_simulations(net, params, root_dir, imgdir, datadir)
        print("training complete.")

        if use_nest:
            nest.ResetKernel()
            print("simulator reset.")

        global_t_processed = time.time() - global_t_start

        t_config = global_t_processed / (i + 1)
        print(f"Avg. time per config: {t_config:.2f}s, " +
              f"ETA: {timedelta(seconds=np.round(t_config * (len(all_configs)-(i+1))))}\n\n")
