import nest
import matplotlib.pyplot as plt
import numpy as np
from networks.network_nest import NestNetwork
from networks.network_numpy import NumpyNetwork
from sklearn.metrics import mean_squared_error as mse
from time import time
import utils
import os
import json
import sys
import argparse
import pandas as pd
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--network",
                    type=str, choices=["numpy", "rnest", "snest"],
                    default="snest",
                    help="""Type of network to train. Choice between exact mathematical simulation ('numpy') and NEST simulations with rate- or spiking neurons ('rnest', 'snest')""")
parser.add_argument("--le",
                    action="store_true",
                    help="""Use latent equilibrium in activation and plasticity."""
                    )
parser.add_argument("--cont",
                    type=str,
                    help="""continue training from a previous simulation""")
parser.add_argument("--weights",
                    type=str,
                    help="Start simulations from a given set of weights to ensure comparable results.")
parser.add_argument("--plot",
                    type=int,
                    default=100,
                    help="generate a plot of training progress after every n epochs.")
parser.add_argument("--mode",
                    type=str,
                    default="bars",
                    help="which dataset to train on")

args = parser.parse_args()



if args.cont:
    root_dir = args.cont
    imgdir = os.path.join(root_dir, "plots")
    datadir = os.path.join(root_dir, "data")
    args.weights = os.path.join(root_dir, "weights.json")
    with open(os.path.join(root_dir, "params.json"), "r") as f:
        all_params = json.load(f)
        sim_params = all_params["simulation"]
        neuron_params = all_params["neurons"]
        syn_params = all_params["synapses"]
        args.le = neuron_params["latent_equilibrium"]
    with open(os.path.join(root_dir, "progress.json"), "r") as f:
        progress = json.load(f)
    spiking = sim_params["spiking"]
else:
    if args.le:
        from params_le import *  # nopep8
    else:
        from params import *  # nopep8
    root_dir, imgdir, datadir = utils.setup_directories(type=args.network)
    spiking = args.network == "snest"
    sim_params["network_type"] = args.network
    sim_params["timestamp"] = root_dir.split(os.path.sep)[-1]
    sim_params["spiking"] = spiking
    # neuron_params["latent_equilibrium"] = args.le



utils.setup_nest(sim_params, datadir)
utils.setup_models(spiking, neuron_params, sim_params, syn_params, False)
if sim_params["network_type"] == "numpy":
    net = NumpyNetwork(sim_params, neuron_params, syn_params, args.mode)
else:
    net = NestNetwork(sim_params, neuron_params, syn_params, args.mode, spiking)

if args.weights:
    with open(args.weights) as f:
        print(f"initializing network with weights from {args.weights}")
        weight_dict = json.load(f)
    net.set_all_weights(weight_dict)

dims = sim_params["dims"]

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)


simulation_times = []
if args.cont:
    net.test_acc = progress["test_acc"]
    net.test_loss = progress["test_loss"]
    net.train_loss = progress["train_loss"]
    net.V_ah_record = np.array(progress["V_ah_record"])
    net.U_y_record = np.array(progress["U_y_record"])
    net.U_i_record = np.array(progress["U_i_record"])
    ff_error = progress["ff_error"]
    fb_error = progress["fb_error"]
    epoch_offset = progress["epochs_completed"]
    net.epoch = epoch_offset
    print(f"continuing training from epoch {epoch_offset}")
else:
    ff_error = []
    fb_error = []
    epoch_offset = 0


if not args.cont:
    # dump simulation parameters and initial weights to .json files
    with open(os.path.join(root_dir, "params.json"), "w") as f:
        json.dump({"simulation": sim_params, "neurons": neuron_params, "synapses": syn_params}, f, indent=4)
    utils.store_synaptic_weights(net, root_dir, "init_weights.json")

print("setup complete, running simulations...")
plot_every = args.plot
start_training = time()

if spiking:
    sr = nest.Create("spike_recorder")
    nest.Connect(nest.GetNodes({"model": neuron_params["model"]}), sr)


try: # catches KeyboardInterruptException to ensure proper cleanup and storage of progress
    t_start_training = time()
    if not args.cont:
        net.test_epoch()
    for epoch in range(epoch_offset, sim_params["n_epochs"] + 1):
        t_start_epoch = time()
        net.train_epoch()
        t_epoch = time() - t_start_epoch
        simulation_times.append(t_epoch)

        if epoch % sim_params["test_interval"] == 0:
            if spiking:
                sr.set({"start": 0, "stop": 8*sim_params["SIM_TIME"], "origin":nest.biological_time, "n_events":0})
            net.test_epoch()
            if spiking:
                spikes = pd.DataFrame.from_dict(sr.events).groupby("senders")
                n_spikes_avg = spikes.count()["times"].mean()
                rate = 1000 *  n_spikes_avg / (8*sim_params["SIM_TIME"])
                print(f"neurons firing at {rate:.1f}Hz")
            
            print(f"test completed, acc: {net.test_acc[-1][1]:.3f}, loss: {net.test_loss[-1][1]:.3f}")
            if epoch > 0:
                t_processed = time() - t_start_training
                t_epoch = t_processed / epoch
                print(f"\tETA: {timedelta(seconds=np.round(t_epoch * (sim_params['n_epochs']-epoch)))}\n")
        
        if plot_every > 0 and epoch % plot_every == 0:
            print(f"plotting epoch {epoch}")

            plt.rcParams['savefig.dpi'] = 300

            fig, axes = plt.subplots(4, 2, constrained_layout=True)
            [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7] = axes.flatten()
            
            intn_error = np.square(net.U_y_record - net.U_i_record)

            mean_error = utils.rolling_avg(np.mean(intn_error, axis=1), size=200)
            ax0.plot(mean_error, color="black")

            intn_error_now = np.mean(mean_error[-20:])
            ax0_2 = ax0.secondary_yaxis("right")
            ax0_2.set_yticks([intn_error_now])

            apical_error = np.linalg.norm(net.V_ah_record, axis=1)
            ax1.plot(utils.rolling_avg(apical_error, size=150))

            apical_error_now = np.mean(apical_error[-20:])
            ax1_2 = ax1.secondary_yaxis("right")
            ax1_2.set_yticks([apical_error_now])

            # Synaptic weights
            # notice that all weights are scaled up again to ensure that derived metrics are comparible between simulations
            weights = net.get_weight_dict()
            WHI = weights[-2]["pi"]
            WHY = weights[-2]["down"]
            WIH = weights[-2]["ip"]
            WYH = weights[-1]["up"]

            fb_error_now = mse(WHY.flatten(), -WHI.flatten())
            fb_error.append([epoch, fb_error_now])


            ff_error_now = mse(WYH.flatten(), WIH.flatten())
            ff_error.append([epoch, ff_error_now])

            ax2.plot(*zip(*fb_error), label=f"FB error: {fb_error_now:.3f}")
            ax3.plot(*zip(*ff_error), label=f"FF error: {ff_error_now:.3f}")

            # plot synaptic weights
            for i in range(dims[2]):
                col = cmap_2(i)
                for j in range(dims[1]):
                    ax4.plot(j, -WHY[j, i], ".", color=col, label=f"to {i}")
                    ax4.plot(j, WHI[j, i], "x", color=col, label=f"from {i}")

            for i in range(dims[1]):
                for j in range(dims[2]):
                    col = cmap_2(j)
                    ax5.plot(i, WYH[j, i], ".", color=col, label=f"to {i}")
                    ax5.plot(i, WIH[j, i], "x", color=col, label=f"from {i}")

            ax6.plot(*zip(*net.test_acc))
            ax7.plot(*zip(*net.test_loss))

            ax0.set_title("interneuron - pyramidal error")
            ax1.set_title("apical error")
            ax2.set_title("Feedback error")
            ax3.set_title("Feedforward error")
            ax4.set_title("Feedback weights")
            ax5.set_title("Feedforward weights")
            ax6.set_title("Test Accuracy")
            ax7.set_title("Test Loss")

            ax0.set_ylim(bottom=0)
            ax1.set_ylim(bottom=0)
            ax2.set_ylim(bottom=0)
            ax3.set_ylim(bottom=0)
            ax6.set_ylim(0, 1)
            ax7.set_ylim(0, 1)

            plt.savefig(os.path.join(imgdir, f"{epoch}.png"))
            plt.close()

            print(f"epoch time: {np.mean(simulation_times[-50:]):.2f}s.")
            print(f"test loss: {net.test_loss[-1][1]:.4f}")
            # print(f"ff error: {ff_error_now:.5f}, fb error: {fb_error_now:.5f}")
            # print(f"apical error: {apical_error_now:.2f}, intn error: {intn_error_now:.4f}\n")

except KeyboardInterrupt:
    print("KeyboardInterrupt received - storing progress...")
finally:
    utils.store_synaptic_weights(net, os.path.dirname(datadir))
    print("Weights stored to disk.")
    progress = {
        "test_acc": net.test_acc,
        "test_loss": net.test_loss,
        "train_loss": net.train_loss,
        "ff_error": ff_error,
        "fb_error": fb_error,
        "V_ah_record": net.V_ah_record.tolist(),
        "U_y_record": net.U_y_record.tolist(),
        "U_i_record": net.U_i_record.tolist(),
        "epochs_completed": epoch
    }
    with open(os.path.join(root_dir, "progress.json"), "w") as f:
        json.dump(progress, f, indent=4)
    print("progress stored to disk, exiting.")
