import argparse
import json
import os
import sys
import warnings
from datetime import timedelta
from time import time

import numpy as np
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params
from src.plot_utils import plot_progress

warnings.simplefilter('error', RuntimeWarning)


def run_simulations(net, params, root_dir, imgdir, datadir, plot_interval=0, progress_interval=200, epoch_offset=0):
    simulation_times = []

    # if spiking:
    #     sr = nest.Create("spike_recorder")
    #     nest.Connect(nest.GetNodes({"model": params.neuron_model}), sr)

    try:  # catches KeyboardInterruptException to ensure proper cleanup and storage of progress
        t_start_training = time()
        net.test_epoch()  # begin with initial test
        for epoch in range(epoch_offset, params.n_epochs + 1):
            t_start_epoch = time()
            net.train_epoch()
            t_epoch = time() - t_start_epoch
            simulation_times.append(t_epoch)

            if epoch % params.test_interval == 0 and params.test_interval > 0:
                # if spiking:
                #     sr.set({"start": 0, "stop": 8*params.sim_time, "origin": nest.biological_time, "n_events": 0})
                net.test_epoch()
                # if spiking:
                #     spikes = pd.DataFrame.from_dict(sr.events).groupby("senders")
                #     n_spikes_avg = spikes.count()["times"].mean()
                #     rate = 1000 * n_spikes_avg / (8*params.sim_time)
                #     print(f"neurons firing at {rate:.1f}Hz")

                # print(f"Ep {epoch}: test completed, acc: {net.test_acc[-1][1]:.3f}, loss: {net.test_loss[-1][1]:.3f}")
                if epoch > 0:
                    t_processed = time() - t_start_training
                    t_epoch = t_processed / epoch

                    current_loss = net.test_loss[-1][1]
                    if current_loss > 10:
                        print("-------------------------------")
                        print(f"extreme output loss recorded ({current_loss}), aborting training progress!")
                        print("-------------------------------\n")

                        break

            if plot_interval > 0 and epoch % plot_interval == 0:
                plot_progress(epoch, net, imgdir)

            if epoch % progress_interval == 0:
                print("storing progress...", end="")
                utils.store_synaptic_weights(net, datadir, f"weights_{epoch}.json")
                # utils.store_progress(net, datadir, epoch)
                print("done.")

            if epoch % 50 == 0:
                print(f"Epoch {epoch} completed: test acc: {net.test_acc[-1][1]:.3f}, loss: {net.test_loss[-1][1]:.3f}")
                print(
                    f"\t epoch time: {np.mean(simulation_times[-50:]):.2f}s, \
ETA: {timedelta(seconds=np.round(t_epoch * (params.n_epochs-epoch)))}\n")

    except KeyboardInterrupt:
        print("KeyboardInterrupt received - storing progress...")
    finally:
        utils.store_synaptic_weights(net, root_dir)
        print("Weights stored to disk.")
        utils.store_progress(net, root_dir, epoch)
        print("progress stored to disk, exiting.")


if __name__ == "__main__":
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
                        default="snest",
                        help="""Type of network to train. Choice between exact mathematical simulation ('numpy') \
and NEST simulations with rate- or spiking neurons ('rnest', 'snest')""")
    parser.add_argument("--cont",
                        type=str,
                        help="""continue training from a previous simulation""")
    parser.add_argument("--weights",
                        type=str,
                        help="Start simulations from a given set of weights to ensure comparable results.")
    parser.add_argument("--plot",
                        type=int,
                        default=0,
                        help="generate a plot of training progress after every n epochs.")
    parser.add_argument("--config",
                        type=str,
                        help="path to a .json file specifying which parameters should deviate from their defaults.")
    parser.add_argument("--threads",
                        type=int,
                        default=10,
                        help="number of threads to allocate. Only has an effect when simulating with NEST.")
    parser.add_argument("--progress",
                        type=int,
                        default=100,
                        help="store training progress after every n steps.")

    args = parser.parse_args()

    if args.cont:
        root_dir = args.cont
        imgdir = os.path.join(root_dir, "plots")
        datadir = os.path.join(root_dir, "data")
        args.weights = os.path.join(root_dir, "weights.json")
        params = Params(os.path.join(root_dir, "params.json"))
        with open(os.path.join(root_dir, "progress.json"), "r") as f:
            progress = json.load(f)
        spiking = params.spiking
    else:
        if args.config:
            params = Params(args.config)
            config_name = os.path.split(args.config)[-1].split(".")[0]
        else:
            params = Params()
            config_name = "default_config"

        root_dir, imgdir, datadir = utils.setup_directories(name=config_name, type=args.network)
        if not root_dir:
            print("a simulation of that name already exists, exiting.")
            sys.exit()

        spiking = args.network == "snest"
        params.network_type = args.network
        params.timestamp = root_dir.split(os.path.sep)[-1]
        params.spiking = spiking
        # params.mode = args.mode
    params.threads = args.threads

    utils.setup_nest(params, datadir)
    if params.network_type == "numpy":
        net = NumpyNetwork(params)
    else:
        net = NestNetwork(params)

    if args.weights:
        with open(args.weights) as f:
            print(f"initializing network with weights from {args.weights}")
            weight_dict = json.load(f)
        net.set_all_weights(weight_dict)

    if args.cont:
        net.test_acc = progress["test_acc"]
        net.test_loss = progress["test_loss"]
        net.train_loss = progress["train_loss"]
        # net.V_ah_record = np.array(progress["V_ah_record"])
        # net.U_y_record = np.array(progress["U_y_record"])
        # net.U_i_record = np.array(progress["U_i_record"])
        # ff_error = progress["ff_error"]
        # fb_error = progress["fb_error"]
        epoch_offset = progress["epochs_completed"]
        net.epoch = epoch_offset
        print(f"continuing training from epoch {epoch_offset}")
    else:
        ff_error = []
        fb_error = []
        epoch_offset = 0

    if not args.cont:
        # dump simulation parameters and initial weights to .json files
        params.to_json(os.path.join(root_dir, "params.json"))
        utils.store_synaptic_weights(net, root_dir, "init_weights.json")

    print("setup complete, running simulations...")
    plot_interval = args.plot
    run_simulations(net, params, root_dir, imgdir, datadir, plot_interval, args.progress, epoch_offset)
