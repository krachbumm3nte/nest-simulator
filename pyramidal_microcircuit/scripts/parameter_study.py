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
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",
                        type=str, choices=["numpy", "rnest", "snest"],
                        help="""Type of network to train. Choice between exact mathematical simulation ('numpy') and \
    NEST simulations with rate- or spiking neurons ('rnest', 'snest')""")
    parser.add_argument("--config_dir",
                        type=str,
                        help="folder in which to search for config files")
    parser.add_argument("--target_dir",
                        type=str,
                        help="directory in which to store")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="overwrite existing simulations using the same configuration")
    parser.add_argument("--threads",
                        type=int,
                        default=8,
                        help="number of threads to allocate. Only has an effect when simulating with NEST.")
    parser.add_argument("--weights",
                        type=str,
                        help="Set of initial weights to be used for all simulations.")


    args = parser.parse_args()

    all_configs = os.listdir(args.config_dir)

    global_t_start = time.time()
    for i, config in enumerate(all_configs):
        print(f"\nprocessing config file {config}...")
        if not config.endswith(".json"):
            print(f"skipping file {config}")
            continue

        params = Params(os.path.join(args.config_dir, config))
        print("created params")
        if params.network_type is None and args.network is None:
            print("no network type specified, aborting.")
            sys.exit()
        else:
            if params.network_type and args.network:
                print(f"both input file and script parameters specify different network types ({params.network_type}/{args.network}).")
                print(f"overwriting with argument and using {args.network} network type")
                params.network_type = args.network
            else:
                print(f"preparing simulation for network type: {params.network_type}")


        spiking = params.network_type == "snest"
        params.spiking = spiking
        params.threads = args.threads

        use_nest = params.network_type != "numpy"

        config_name = os.path.split(config)[-1].split(".")[0]
        root_dir, imgdir, datadir = utils.setup_directories(name=config_name, type=params.network_type, root=args.target_dir)
        if not root_dir:
            print("\ta simulation of that name already exists, skipping.\n")
            continue
        print(f"created dirs: {root_dir}")

        if not use_nest:
            net = NumpyNetwork(params)
        else:
            utils.setup_nest(params, datadir)
            net = NestNetwork(params)

        if args.weights:
            with open(args.weights, "r") as f:
                weight_dict = json.load(f)
            print(f"setting network weights from file: {args.weights}")
            net.set_all_weights(weight_dict)


        simulation_times = []

        # dump simulation parameters and initial weights to .json files
        params.to_json(os.path.join(root_dir, "params.json"))
        utils.store_synaptic_weights(net, root_dir, "init_weights.json")

        print(f"Setup complete, beginning to train...")

        run_simulations(net, params, root_dir, imgdir, datadir)

        print("training complete.")
        if use_nest:
            nest.ResetKernel()
        print("simulator reset.")
        global_t_processed = time.time() - global_t_start

        t_config = global_t_processed / (i + 1)
        print(
            f"time per training: {t_config:.2f}s, ETA: {timedelta(seconds=np.round(t_config * (len(all_configs)-i)))}\n\n")
