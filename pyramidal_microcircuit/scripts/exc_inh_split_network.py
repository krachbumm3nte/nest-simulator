import argparse
import json
import os
import sys
import warnings
from datetime import timedelta
from time import time
import nest
import numpy as np
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params
from src.plot_utils import plot_training_progress, plot_pre_training
from copy import deepcopy
warnings.simplefilter('error', RuntimeWarning)


def run_simulations(net, params, root_dir, imgdir, datadir, plot_interval=0, progress_interval=200, epoch_offset=0):
    simulation_times = []

    try:  # catches KeyboardInterruptException to ensure proper cleanup and storage of progress
        t_start_training = time()
        net.test_epoch()  # begin with initial test
        for epoch in range(epoch_offset, params.n_epochs + 1):
            t_start_epoch = time()
            net.train_epoch()
            t_epoch = time() - t_start_epoch
            simulation_times.append(t_epoch)

            if epoch % params.test_interval == 0 and params.test_interval > 0:
                net.test_epoch()

                if epoch > 0:
                    current_loss = net.test_loss[-1][1]
                    if current_loss > 5000:
                        print("-------------------------------")
                        print(f"extreme output loss recorded ({current_loss}), aborting training progress!")
                        print("-------------------------------\n")
                        break

            if plot_interval > 0 and epoch % plot_interval == 0:
                if net.mode == "self-pred":
                    plot_pre_training(epoch, net, os.path.join(imgdir, f"{epoch}.png"))
                else:
                    plot_training_progress(epoch, net, os.path.join(imgdir, f"{epoch}.png"))

            if epoch % progress_interval == 0:
                print("storing progress...", end="")
                utils.store_synaptic_weights(net, os.path.join(datadir, f"weights_{epoch}.json"))
                utils.store_progress(net, root_dir, epoch)
                print("done.")

            if epoch % 5 == 0:
                v_intn = np.array([e["V_m"] for e in net.layers[0].intn.get("soma")])
                v_intn_2 = np.array([e["V_m"] for e in net.layers[0].intn_2.get("soma")])
                print(v_intn, v_intn_2, v_intn-v_intn_2)
                if net.mode == "self-pred":
                    print(
                        f"Epoch {epoch} completed: intn error: {net.intn_error[-1][1]:.3f}, apical error: {net.apical_error[-1][1]:.3f}")
                else:
                    print(
                        f"Epoch {epoch} completed: test acc: {net.test_acc[-1][1]:.3f}, loss: {net.test_loss[-1][1]:.3f}")
                print(f"\t epoch time: {np.mean(simulation_times[-50:]):.2f}s, \
ETA: {timedelta(seconds=np.round(t_epoch * (params.n_epochs-epoch)))}\n")

    except KeyboardInterrupt:
        print("KeyboardInterrupt received - storing progress...")
    finally:
        utils.store_synaptic_weights(net, os.path.join(root_dir, "weights.json"))
        print("Weights stored to disk.")
        utils.store_progress(net, root_dir, epoch)
        print("progress stored to disk, exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",
                        type=str, choices=["numpy", "rnest", "snest"],
                        help="""Type of network to train. Choice between exact mathematical simulation ('numpy') and \
    NEST simulations with rate- or spiking neurons ('rnest', 'snest')""")
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

        if not args.network and not params.network_type:
            print("no network type specified, aborting.")
            sys.exit()
        if params.network_type and args.network and args.network != params.network_type:
            print(
                f"both input file and script parameters specify different \
network types ({params.network_type}/{args.network}).")
            print(f"overwriting with argument and using {args.network} network type")
            params.network_type = args.network

        spiking = params.network_type == "snest"
        params.spiking = spiking
        root_dir, imgdir, datadir = utils.setup_directories(name=config_name, type=params.network_type)
        # params.mode = args.mode
    params.threads = args.threads

    utils.setup_nest(params, datadir)

    init_weights = None
    if args.weights:
        print(f"initializing network with weights from {args.weights}")
        with open(args.weights) as f:
            init_weights = json.load(f)

    net = NestNetwork(params)

    for l in net.layers[:-1]:
        n_pyr = l.N_pyr
        n_intn = l.N_next

        n_intn_inh = l.N_next
        # Modify existing connection to allow strictly positive weights
        w_exc_1 = NestNetwork.gen_weights(n_intn, n_pyr, 0, params.wmax_init)
        l.pi.set({"Wmin": 0, "delay": 2*params.delta_t})
        net.set_weights_from_syn(w_exc_1, l.pi)

        # Create inhibitory interneuron population
        # TODO: how much does population size matter here?
        intn_inh_params = deepcopy(params.intn_params)
        intn_inh_params["use_phi"] = False
        intn_inh_params["latent_equilibrium"] = False
        intn_inh_params["soma"]["g"] = 1  # somatic leakage conductance

        l.intn_2 = nest.Create(params.neuron_model, n_intn_inh, intn_inh_params)
        # l.intn_2 = nest.Create("parrot_neuron", n_intn_inh)

        # Connect excitatory interneurons to inhibitory interneuron population
        # syn_spec_exc_2 = deepcopy(l.synapses["down"])
        # syn_spec_exc_2["weight"] = NestNetwork.gen_weights(n_intn, n_intn_inh, 0, 0.5*params.Wmax)
        syn_spec_exc_2 = deepcopy(params.syn_static)
        syn_spec_exc_2["weight"] = 1  # np.random.random(n_intn_inh) * 0.5*params.Wmax
        syn_spec_exc_2["receptor_type"] = params.compartments["soma"]
        nest.Connect(l.intn, l.intn_2, conn_spec="one_to_one", syn_spec=syn_spec_exc_2)

        # Connect inhibitory interneurons to pyramidal targets.
        syn_spec_w_inh = deepcopy(l.synapses["pi"])
        syn_spec_w_inh["weight"] = NestNetwork.gen_weights(n_intn_inh, n_pyr, -params.wmax_init, 0)
        syn_spec_w_inh["Wmax"] = 0
        nest.Connect(l.intn_2, l.pyr, conn_spec="all_to_all", syn_spec=syn_spec_w_inh)

    if args.cont:
        net.test_acc = progress["test_acc"]
        net.test_loss = progress["test_loss"]
        net.train_loss = progress["train_loss"]
        net.ff_error = progress["ff_error"]
        net.fb_error = progress["fb_error"]
        epoch_offset = progress["epochs_completed"]
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
