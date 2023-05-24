# -*- coding: utf-8 -*-
#
# train_network.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import json
import os
import sys
from datetime import timedelta
from time import time

import numpy as np
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params
from src.plot_utils import plot_pre_training, plot_training_progress


def run_simulations(net, params, root_dir, imgdir, datadir, plot_interval=0, progress_interval=200, epoch_offset=0):
    """Trains a network according to a given set of parameters

    Arguments:
        net -- instance of networks.Network
        params -- instance of params.Params
        root_dir -- base dir for storing simulation-relevant files
        imgdir -- subdirectory for plots
        datadir -- subdirectory for intermittent weight files

    Keyword Arguments:
        plot_interval -- number of training epochs after which to generate a figure of training progress (default: {0})
        progress_interval -- number of training epochs after which to store training progress to disk (default: {200})
        epoch_offset -- offset for indexing training data when continuing previous simulations (default: {0})
    """
    simulation_times = []

    try:  # catches KeyboardInterruptException to ensure proper cleanup and storage of progress upon abort
        if epoch_offset == 0:
            net.test_epoch()  # begin each simulation with an initial test

        # core training loop
        for epoch in range(epoch_offset, params.n_epochs + 1):
            t_start_epoch = time()
            net.train_epoch()
            t_epoch = time() - t_start_epoch
            simulation_times.append(t_epoch)

            # perform tests
            if epoch % params.test_interval == 0 and params.test_interval > 0:
                net.test_epoch()

                if epoch > 0:
                    # Under some circumstances, gradients and weights explode, breaking the network and vastly
                    # increasing simulation time. Current approach is to abort training altogether.
                    current_loss = net.test_loss[-1][1]
                    if current_loss > 5000:
                        print("-------------------------------")
                        print(f"extreme output loss recorded ({current_loss}), aborting training progress!")
                        print("-------------------------------\n")
                        break

                if net.mode == "self-pred":
                    print(f"Epoch {epoch}: intn error: {net.intn_error[-1][1]:.3f}," +
                          f"apical error: {net.apical_error[-1][1]:.3f}")
                else:
                    print(f"Epoch {epoch}: test acc: {net.test_acc[-1][1]:.3f}, " +
                          f"test loss: {net.test_loss[-1][1]:.5f}, train loss: {net.train_loss[-1][1]:.5f}")

                print(f"\t epoch time: {np.mean(simulation_times[-50:]):.2f}s, " +
                      f"ETA: {timedelta(seconds=np.round(t_epoch * (params.n_epochs-epoch)))}\n")

            # plot progress
            if plot_interval > 0 and epoch % plot_interval == 0:
                if net.mode == "self-pred":
                    plot_pre_training(epoch, net, os.path.join(imgdir, f"{epoch}.png"))
                else:
                    plot_training_progress(epoch, net, os.path.join(imgdir, f"{epoch}.png"))

            # update .json file with current loss and accuracy scores.
            if epoch % progress_interval == 0:
                print("storing progress...", end="")
                utils.store_synaptic_weights(net, os.path.join(datadir, f"weights_{epoch}.json"))
                utils.store_progress(net, root_dir, epoch)
                print("Done.")

    except KeyboardInterrupt:
        print("KeyboardInterrupt received - storing progress...")
    finally:
        utils.store_synaptic_weights(net, os.path.join(root_dir, "weights.json"))
        print("Weights stored to disk.")
        utils.store_progress(net, root_dir, epoch)
        print("progress stored to disk.")

        if net.mode == "self-pred":
            plot_pre_training(epoch, net, os.path.join(imgdir, f"{epoch}.png"))
        else:
            plot_training_progress(epoch, net, os.path.join(imgdir, f"{epoch}.png"))
        print("progression plot stored to disk.")
        print("Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_network", usage="General training script for dendritic error " +
                                     "networks. If run without arguments, a spiking neural network is trained " +
                                     "on the 'Bars' dataset. Most parameters should be set in a .json file " +
                                     "located at --config.")
    parser.add_argument("--network",
                        type=str, choices=["numpy", "rnest", "snest"],
                        help="Type of network to train. Choice between matrix-based simulation  of rate neurons " +
                        "('numpy') and NEST simulations with rate- or spiking neurons ('rnest', 'snest') ")
    parser.add_argument("--cont",
                        type=str,
                        help="continue training of the simulation at the specified location. Ignores some " +
                        "other arguments")
    parser.add_argument("--weights",
                        type=str,
                        help="Start simulations from a set of weights stored in the specified .json file.")
    parser.add_argument("--plot",
                        type=int,
                        default=0,
                        help="generate a plot of training progress after every n epochs.")
    parser.add_argument("--config",
                        type=str,
                        help="path to a .json file specifying which parameters should deviate from their defaults.")
    parser.add_argument("--threads",
                        type=int,
                        default=8,
                        help="number of threads to allocate. Only has an effect when simulating with NEST.")
    parser.add_argument("--progress",
                        type=int,
                        default=100,
                        help="store training progress after every n steps.")

    args = parser.parse_args()

    if args.cont:
        # In this special case, network is set up so as to mirror the
        # final iteration of a previous run exactly. This becomes somewhat
        # involved, but functions correctly to the best of my knowledge.
        root_dir = args.cont
        imgdir = os.path.join(root_dir, "plots")
        datadir = os.path.join(root_dir, "data")

        # This is somewhat ugly, but reading out the full params file created by
        # a network leads to some errors that are annoying to fix. So the
        # location of the original config file is read out, and the network is
        # instantiated from that instead.
        args.weights = os.path.join(root_dir, "weights.json")
        with open(os.path.join(root_dir, "params.json")) as f:
            p_full = json.load(f)
        params_dir = p_full["config_file"]

        basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        params = Params(os.path.join(basedir, params_dir))
        with open(os.path.join(root_dir, "progress.json"), "r") as f:
            progress = json.load(f)

        params.init_self_pred = False
        spiking = params.spiking
    else:
        if args.config:
            params = Params(args.config)
            config_name = os.path.split(args.config)[-1].split(".")[0]
        else:
            params = Params()
            config_name = "default_config"

        if not args.network and not params.network_type:
            print("no network type specified, aborting.")
            sys.exit()
        if params.network_type and args.network and args.network != params.network_type:
            print(f"both input file and script parameters specify different" +
                  f"network types ({params.network_type}/{args.network}).")

            print(f"overwriting with argument and using {args.network} network type")
            params.network_type = args.network

        spiking = params.network_type == "snest"
        params.spiking = spiking
        root_dir, imgdir, datadir = utils.setup_directories(name=config_name, type=params.network_type)

    params.threads = args.threads

    utils.setup_nest(params, datadir)

    init_weights = None
    if args.weights:
        print(f"initializing network with weights from {args.weights}")
        with open(args.weights) as f:
            init_weights = json.load(f)

    if params.network_type == "numpy":
        net = NumpyNetwork(params)
        if init_weights:
            net.set_all_weights(init_weights)  # TODO: unify weight initialization
    else:
        net = NestNetwork(params, init_weights)

    if args.cont:
        # If continuing previous training, read out progress and store it in the
        # network class again.
        net.test_acc = progress["test_acc"]
        net.test_loss = progress["test_loss"]
        net.train_loss = progress["train_loss"]
        net.ff_error = progress["ff_error"]
        net.fb_error = progress["fb_error"]
        epoch_offset = progress["epochs_completed"] + 1
        net.epoch = epoch_offset
        print(f"continuing training from epoch {epoch_offset}")
    else:
        epoch_offset = 0

    if not args.cont:
        # dump simulation parameters and initial weights to .json files
        params.to_json(os.path.join(root_dir, "params.json"))

        init_weight_fp = os.path.join(root_dir, "init_weights.json")
        if init_weights:
            with open(init_weight_fp, "w") as f:
                json.dump(init_weights, f, indent=4)
        else:
            utils.store_synaptic_weights(net, init_weight_fp)

    print("setup complete, running simulations...")
    plot_interval = args.plot
    run_simulations(net, params, root_dir, imgdir, datadir, plot_interval, args.progress, epoch_offset)
