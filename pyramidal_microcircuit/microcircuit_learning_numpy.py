import matplotlib.pyplot as plt
import numpy as np
from params import *
import pandas as pd
from networks.network_numpy import NumpyNetwork
from sklearn.metrics import mean_squared_error as mse
from time import time
import utils
import os
import json

root, imgdir, datadir = utils.setup_simulation()

utils.setup_models(False, neuron_params, sim_params, syn_params, False)


net = NumpyNetwork(sim_params, neuron_params, syn_params)

dims = sim_params["dims"]

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

plot_interval = 50

T = []
ff_errors = []
fb_errors = []
apical_errors = []
intn_errors = [[] for i in range(dims[-1])]


# dump simulation parameters to a .json file
with open(os.path.join(os.path.dirname(imgdir), "params.json"), "w") as f:
    # print(sim_params, neuron_params, syn_params)
    for conn in ["hx", "yh", "hy", "ih", "hi"]:
        syn_params[conn]["weight"] = net.conns[conn]["w"].tolist()
    json.dump({"simulation": sim_params, "neurons": neuron_params, "synapses": syn_params}, f)
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

            mean_error = utils.rolling_avg(np.sum(intn_error, axis=1), size=200)
            abs_voltage = np.mean(np.concatenate([net.U_y_record[-5:], net.U_i_record[-5:]]))
            ax0.plot(mean_error, color="black")

            intn_error_now = np.mean(mean_error[-20:])
            ax0_2 = ax0.secondary_yaxis("right")
            ax0_2.set_yticks([intn_error_now])

            # plot apical error
            apical_error = utils.rolling_avg(np.linalg.norm(net.V_ah_record, axis=1), size=150)
            ax1.plot(apical_error)

            apical_error_now = np.mean(apical_error[-20:])
            ax1_2 = ax1.secondary_yaxis("right")
            ax1_2.set_yticks([apical_error_now])

            # Synaptic weights
            WHY = net.conns["hy"]["w"]
            WHI = net.conns["hi"]["w"]
            WYH = net.conns["yh"]["w"]
            WIH = net.conns["ih"]["w"]

            fb_error = mse(WHY.flatten(), -WHI.flatten())
            fb_errors.append(fb_error)
            ax2.plot(fb_errors, label=f"FB error: {fb_error:.3f}")

            ff_error = mse(WYH.flatten(), WIH.flatten())
            ff_errors.append(ff_error)
            ax3.plot(ff_errors, label=f"FF error: {ff_error:.3f}")

            # plot weights
            for i in range(dims[2]):
                col = cmap_2(i)
                for j in range(dims[1]):
                    ax4.plot(j, -WHY[j, i], ".", color=col, label=f"to {t}")
                    ax4.plot(j, WHI[j, i], "x", color=col, label=f"from {t}")

            for i in range(dims[1]):
                for j in range(dims[2]):
                    col = cmap_2(j)
                    ax5.plot(i, WYH[j, i], ".", color=col, label=f"to {t}")
                    ax5.plot(i, WIH[j, i], "x", color=col, label=f"from {t}")

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
                f"ff error: {ff_error:.5f}, fb error: {fb_error:.5f}, interneuron error: {intn_error_now:.4f}, absolute somatic voltage: {abs_voltage:.3f}\n")

        elif run % 100 == 0:
            print(f"run {run} completed.")
except KeyboardInterrupt:
    print("KeyboardInterrupt received - storing synaptic weights...")
finally:
    utils.store_synaptic_weights(net, os.path.dirname(datadir))
    print("weights stored to disk, exiting.")
