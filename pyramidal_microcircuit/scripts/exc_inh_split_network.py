# -*- coding: utf-8 -*-
#
# exc_inh_split_network.py
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
from copy import deepcopy

import numpy as np
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.params import Params
from train_network import run_simulations

import nest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="exc_inh_split_network", usage="A variant of train_network.py, in which " +
                                     "lateral intn->pyr connections conform to dale's law.")
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
        raise NotImplementedError("Not gonna happen...")
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

    net = NestNetwork(params)

    for layer in net.layers[:-1]:
        n_pyr = layer.N_pyr
        n_intn = layer.N_next

        n_intn_inh = 4 * n_pyr
        # Modify existing connection to allow strictly positive weights
        w_exc_1 = net.gen_weights(n_intn, n_pyr, 0, params.wmax_init)
        layer.pi.set({"Wmin": 0, "delay": 2*params.delta_t})
        net.set_weights_from_syn(w_exc_1, layer.pi)

        # Create inhibitory interneuron population
        # TODO: how much does population size matter here?
        intn_inh_params = deepcopy(params.intn_params)
        intn_inh_params["use_phi"] = False
        intn_inh_params["latent_equilibrium"] = False
        intn_inh_params["soma"]["g"] = 1  # somatic leakage conductance

        layer.intn_2 = nest.Create(params.neuron_model, n_intn_inh, intn_inh_params)
        # l.intn_2 = nest.Create("parrot_neuron", n_intn_inh)
        weight_factor = 1.5

        # Connect excitatory interneurons to inhibitory interneuron population
        # syn_spec_exc_2 = deepcopy(l.synapses["down"])
        syn_spec_exc_2 = deepcopy(params.syn_static)
        syn_spec_exc_2["weight"] = 1  # np.random.random(n_intn_inh) * 0.5*params.Wmax
        syn_spec_exc_2["weight"] = net.gen_weights(n_intn, n_intn_inh, 0, params.wmax_init/n_pyr)
        syn_spec_exc_2["receptor_type"] = params.compartments["soma"]
        layer.syn_exc_2 = nest.Connect(layer.intn, layer.intn_2, conn_spec="all_to_all",
                                       syn_spec=syn_spec_exc_2, return_synapsecollection=True)

        # Connect inhibitory interneurons to pyramidal targets.
        syn_spec_w_inh = deepcopy(layer.synapses["pi"])
        syn_spec_w_inh["weight"] = net.gen_weights(n_intn_inh, n_pyr, -weight_factor*params.wmax_init/n_intn_inh, 0)
        syn_spec_w_inh["Wmax"] = 0
        layer.syn_inh = nest.Connect(layer.intn_2, layer.pyr, conn_spec="all_to_all",
                                     syn_spec=syn_spec_w_inh, return_synapsecollection=True)

        dropout = 0.3
        if dropout > 0:
            indices = np.random.choice(n_intn_inh, round(dropout * n_intn_inh), replace=False)
            for i in indices:
                nest.Disconnect(layer.syn_exc_2[i])

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
