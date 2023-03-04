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
classes = {
    "rate": [
        "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/runs/numpy_network_success",
    ],
    "NEST rate": [
        # "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/runs/rate_learning_nest",
    ],
    "NEST spiking": [
        "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/runs/full_run_spiking",
        "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/runs/full_run_spiking_2",
    ]
}

fig, axes = plt.subplots(2, 3, sharex=True)


for name, data_dirs in classes.items():
    test_acc = []
    test_loss = []
    train_loss = []
    apical_error = []
    U_y_record = []
    U_i_record = []
    ff_error = []
    fb_error = []

    for root_dir in data_dirs:
        imgdir = os.path.join(root_dir, "plots")
        datadir = os.path.join(root_dir, "data")

        with open(os.path.join(root_dir, "weights.json")) as f:
            post_training_weights = json.load(f)
        with open(os.path.join(root_dir, "params.json"), "r") as f:
            all_params = json.load(f)
            sim_params = all_params["simulation"]
            neuron_params = all_params["neurons"]
            syn_params = all_params["synapses"]
        with open(os.path.join(root_dir, "progress.json"), "r") as f:
            progress = json.load(f)

        test_acc.append(progress["test_acc"])
        test_loss.append(progress["test_loss"])
        train_loss.append(progress["train_loss"])
        # apical_error.append(np.linalg.norm(np.array(progress["V_ah_record"]), axis=1))
        # U_y_record.append(np.array(progress["U_y_record"]))
        # U_i_record.append(np.array(progress["U_i_record"]))
        ff_error.append(progress["ff_error"])
        fb_error.append(progress["fb_error"])

    test_acc = np.mean(test_acc, axis=0)
    test_loss = np.mean(test_loss, axis=0)
    train_loss = np.mean(train_loss, axis=0)
    # apical_error = np.mean(apical_error, axis=0)
    # U_y_record = np.mean(U_y_record, axis=0)
    # U_i_record = np.mean(U_i_record, axis=0)
    ff_error = np.mean(ff_error, axis=0)
    # fb_error = np.mean(fb_error, axis=0)

    axes[0][0].plot(*zip(*test_acc), label=name)
    axes[0][1].plot(*zip(*test_loss), label=name)
    axes[1][0].plot(*zip(*train_loss), label=name)
    axes[1][1].plot(ff_error, label=name)


    axes[0][0].legend("test Accuracy")
    axes[0][1].legend("test Loss")
    axes[1][0].legend("train loss")
    axes[1][1].legend("FF weight error")

plt.show()





if not args.cont:
    # dump simulation parameters and initial weights to .json files
    with open(os.path.join(root_dir, "params.json"), "w") as f:
        json.dump({"simulation": sim_params, "neurons": neuron_params, "synapses": syn_params}, f, indent=4)
    utils.store_synaptic_weights(net, root_dir, "init_weights.json")

print("setup complete, running simulations...")
net.test_bars()
try:
    for epoch in range(sim_params["n_epochs"] + 1):
        start = time()
        net.train_epoch_bars()
        t = time() - start
        T.append(t)

        if epoch % plot_interval == 0:
            print(f"plotting epoch {epoch}")
            start = time()
            net.test_bars()
            fig, axes = plt.subplots(4, 2, constrained_layout=True)
            [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7] = axes.flatten()
            plt.rcParams['savefig.dpi'] = 300

            intn_error = np.square(net.U_y_record - net.U_i_record)

            mean_error = utils.rolling_avg(np.mean(intn_error, axis=1), size=200)
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
            WHI = weights[-2]["pi"]
            WHY = weights[-2]["down"]
            WIH = weights[-2]["ip"]
            WYH = weights[-1]["up"]

            fb_error_now = mse(WHY.flatten(), -WHI.flatten())
            fb_error.append([epoch, fb_error_now])
            ax2.plot(*zip(*fb_error), label=f"FB error: {fb_error_now:.3f}")

            ff_error_now = mse(WYH.flatten(), WIH.flatten())
            ff_error.append([epoch, ff_error_now])
            ax3.plot(*zip(*ff_error), label=f"FF error: {ff_error_now:.3f}")

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

            plot_duration = time() - start
            print(f"sim time: {np.mean(T[-50:]):.2f}s. plot time:{plot_duration:.2f}s.")
            print(f"train loss: {net.train_loss[-1][1]:.4f}, test loss: {net.test_loss[-1][1]:.4f}")
            print(f"ff error: {ff_error_now:.5f}, fb error: {fb_error_now:.5f}")
            print(f"apical error: {apical_error_now:.2f}, intn error: {intn_error_now:.4f}\n")

        elif epoch % 100 == 0:
            print(f"epoch {epoch} completed.")
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
        "U_i_record": net.U_i_record.tolist()
    }
    with open(os.path.join(root_dir, "progress.json"), "w") as f:
        json.dump(progress, f, indent=4)
    print("progress stored to disk, exiting.")
