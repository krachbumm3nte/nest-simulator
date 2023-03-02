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

parser = argparse.ArgumentParser()
parser.add_argument("--network",
                    type=str, choices=["numpy", "rnest", "snest"],
                    default="numpy",
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
args = parser.parse_args()
print(args.le)
if args.le:
    from params_le import *  # nopep8
else:
    from params import *  # nopep8


neuron_params["latent_equilibrium"] = args.le

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
    with open(os.path.join(root_dir, "progress.json"), "r") as f:
        progress = json.load(f)
else:
    root_dir, imgdir, datadir = utils.setup_directories()


spiking = args.network == "snest"
sim_params["spiking"] = spiking
sim_params["network_type"] = args.network

utils.setup_nest(sim_params, datadir)
utils.setup_models(spiking, neuron_params, sim_params, syn_params, False)
if args.network == "numpy":
    net = NumpyNetwork(sim_params, neuron_params, syn_params)
else:
    net = NestNetwork(sim_params, neuron_params, syn_params, spiking)
    print(nest.GetConnections())
    print(nest.GetNodes())


if args.weights:
    with open(args.weights) as f:
        weight_dict = json.load(f)
    net.set_weights(weight_dict)

dims = sim_params["dims"]

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

plot_interval = 50

T = []

if args.cont:
    net.test_acc = progress["test_acc"]
    net.test_loss = progress["test_loss"]
    net.train_loss = progress["train_loss"]
    net.V_ah_record = np.array(progress["V_ah_record"])
    net.U_y_record = np.array(progress["U_y_record"])
    net.U_i_record = np.array(progress["U_i_record"])
    ff_error = progress["ff_error"]
    fb_error = progress["fb_error"]
else:
    ff_error = []
    fb_error = []

if not args.cont:
    # dump simulation parameters and initial weights to .json files
    with open(os.path.join(root_dir, "params.json"), "w") as f:
        json.dump({"simulation": sim_params, "neurons": neuron_params, "synapses": syn_params}, f)
    utils.store_synaptic_weights(net, root_dir, "init_weights.json")

print("setup complete, running simulations...")

try:
    for run in range(sim_params["n_runs"] + 1):
        start = time()
        net.train_epoch_bars()
        t = time() - start
        T.append(t)

        if run % plot_interval == 0:
            net.test_bars()
            print(f"plotting run {run}")
            start = time()
            fig, axes = plt.subplots(4, 2, constrained_layout=True)
            [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7] = axes.flatten()
            plt.rcParams['savefig.dpi'] = 300

            intn_error = np.square(net.U_y_record - net.U_i_record)

            mean_error = utils.rolling_avg(np.mean(intn_error, axis=1), size=200)
            abs_voltage = np.mean(np.concatenate([net.U_y_record[-5:], net.U_i_record[-5:]]))
            ax0.plot(mean_error, color="black")

            intn_error_now = np.mean(mean_error[-20:])
            ax0_2 = ax0.secondary_yaxis("right")
            ax0_2.set_yticks([intn_error_now])

            # plot apical error
            apical_error = np.linalg.norm(net.V_ah_record, axis=1)
            ax1.plot(utils.rolling_avg(apical_error, size=150))

            apical_error_now = np.mean(apical_error[-20:])
            ax1_2 = ax1.secondary_yaxis("right")
            ax1_2.set_yticks([apical_error_now])

            # Synaptic weights
            # notice that all weights are scaled up again to ensure that derived metrics are comparible between simulations
            weights = net.get_weight_dict()
            WYH = weights[-1]["up"]
            WHI = weights[-2]["pi"]
            WHY = weights[-2]["down"]
            WIH = weights[-2]["ip"]

            fb_error_now = mse(WHY.flatten(), -WHI.flatten())
            fb_error.append(fb_error_now)
            ax2.plot(fb_error, label=f"FB error: {fb_error_now:.3f}")

            ff_error_now = mse(WYH.flatten(), WIH.flatten())
            ff_error.append(ff_error_now)
            ax3.plot(ff_error, label=f"FF error: {ff_error_now:.3f}")

            # plot weights
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

            ax6.plot(utils.rolling_avg(net.test_acc, 2))
            ax7.plot(utils.rolling_avg(net.test_loss, 2))

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

            plt.savefig(os.path.join(imgdir, f"{run}.png"))
            plt.close()

            plot_duration = time() - start
            print(f"mean simulation time: {np.mean(T[-50:]):.4f}s. plot time:{plot_duration:.2f}s. \
    apical error: {apical_error_now:.2f}, train loss: {net.train_loss[-1]:.4f}, test loss: {net.test_loss[-1]:.4f}")
            print(
                f"ff error: {ff_error_now:.5f}, fb error: {fb_error_now:.5f}, interneuron error: {intn_error_now:.4f}, absolute somatic voltage: {abs_voltage:.3f}\n")

        elif run % 100 == 0:
            print(f"run {run} completed.")
except KeyboardInterrupt:
    print("KeyboardInterrupt received - storing synaptic weights...")
finally:
    utils.store_synaptic_weights(net, os.path.dirname(datadir))
    progress = {
        "test_acc": net.test_acc,
        "test_loss": net.test_loss,
        "train_loss": net.train_loss,
        "ff_error": ff_error,
        "fb_error": fb_error,
        "V_ah_record": net.V_ah_record.tolist(),
        "U_y_record": net.U_y_record.tolist(),
        "U_i_record": net.U_i_record.tolist()
    }
    with open(os.path.join(root_dir, "progress.json"), "w") as f:
        json.dump(progress, f)
    print("weights stored to disk, exiting.")
