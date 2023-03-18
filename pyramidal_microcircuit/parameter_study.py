import sys
import argparse
import os
from networks.params import Params
import utils
from networks.network_nest import NestNetwork
from networks.network_numpy import NumpyNetwork
import time
import nest
from microcircuit_learning import run_simulations
import numpy as np
from datetime import timedelta


parser = argparse.ArgumentParser()
# parser.add_argument("--le",
#                     action="store_true",
#                     help="""Use latent equilibrium in activation and plasticity."""
#                     )
# parser.add_argument("--mode",
#                     type=str,
#                     default="bars",
#                     help="which dataset to train on")
parser.add_argument("--network",
                    type=str, choices=["numpy", "rnest", "snest"],
                    default="rnest",
                    help="""Type of network to train. Choice between exact mathematical simulation ('numpy') and NEST simulations with rate- or spiking neurons ('rnest', 'snest')""")
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

args = parser.parse_args()


all_configs = os.listdir(args.config_dir)

global_t_start = time.time()
for i, config in enumerate(all_configs):
    print(f"\nprocessing config file {config}...")
    if not config.endswith(".json"):
        print(f"skipping file {config}")
        continue
    nest.ResetKernel()

    config_name = os.path.split(config)[-1].split(".")[0]
    root_dir, imgdir, datadir = utils.setup_directories(name=config_name, type=args.network)
    if not root_dir:
        print("\ta simulation of that name already exists, skipping.\n")
        continue

    params = Params(os.path.join(args.config_dir, config))
    spiking = args.network == "snest"
    params.network_type = args.network
    params.timestamp = root_dir.split(os.path.sep)[-1]
    params.spiking = spiking
    params.threads = args.threads

    utils.setup_nest(params, datadir)
    if params.network_type == "numpy":
        net = NumpyNetwork(params)
    else:
        net = NestNetwork(params)

    simulation_times = []

    # dump simulation parameters and initial weights to .json files
    params.to_json(os.path.join(root_dir, "params.json"))
    utils.store_synaptic_weights(net, root_dir, "init_weights.json")

    print(f"simulation set up. Start training for {config_name}...")

    run_simulations(net, params, root_dir, imgdir, datadir)

    print("training complete.")
    global_t_processed = time.time() - global_t_start

    t_config = global_t_processed / i
    print(
        f"time per training: {t_config:.2f}s, ETA: {timedelta(seconds=np.round(t_config * (len(all_configs)-i)))}\n\n")
