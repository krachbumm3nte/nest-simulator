import nest
import matplotlib.pyplot as plt
import numpy as np
from params import *
import pandas as pd
from networks.network_nest import NestNetwork
from sklearn.metrics import mean_squared_error as mse
from time import time
import utils
import os
import json

root_dir, imgdir, datadir = utils.setup_simulation()
utils.setup_nest(sim_params, datadir)

spiking = True
utils.setup_models(spiking, neuron_params, sim_params, syn_params, False)

net = NestNetwork(sim_params, neuron_params, syn_params, spiking)

dims = sim_params["dims"]

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

plot_interval = 50

T = []
ff_errors = []
fb_errors = []
apical_errors = []
intn_errors = [[] for i in range(dims[-1])]

# storing populations and their ids for easy access during plotting.
in_ = net.pyr_pops[0]
in_id = sorted(in_.get("global_id"))

hidden = net.pyr_pops[1]
hidden_id = sorted(hidden.get("global_id"))

out = net.pyr_pops[2]
out_id = sorted(out.get("global_id"))

interneurons = net.intn_pops[0]
intn_id = sorted(interneurons.get("global_id"))

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
            if run % plot_interval == 0:
                net.test_bars()

            print(f"plotting run {run}")
            start = time()
            fig, axes = plt.subplots(4, 2, constrained_layout=True)
            [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7] = axes.flatten()
            plt.rcParams['savefig.dpi'] = 300

            # plot somatic voltages of hidden interneurons and output pyramidal neurons
            neuron_data = utils.read_data(net.mm.global_id, datadir, it_min=run-plot_interval+1)

            U_I = neuron_data[neuron_data.sender.isin(intn_id)].groupby("sender")["V_m.s"]
            U_Y = neuron_data[neuron_data.sender.isin(out_id)].groupby("sender")["V_m.s"]

            abs_voltage = []
            for idx, (intn, pyr) in enumerate(zip(intn_id, out_id)):
                data_intn = U_I.get_group(intn)
                data_pyr = U_Y.get_group(pyr)
                int_v = data_intn.array
                pyr_v = data_pyr.array

                abs_voltage.append(np.abs(int_v))
                abs_voltage.append(np.abs(pyr_v))
                error = np.square(int_v-pyr_v)
                intn_errors[idx].extend(error)

            mean_error = utils.rolling_avg(np.mean(intn_errors, axis=0), size=200)
            abs_voltage = np.mean(abs_voltage)
            ax0.plot(mean_error, color="black")

            intn_error_now = np.mean(mean_error[-20:])
            ax0_2 = ax0.secondary_yaxis("right")
            ax0_2.set_yticks([intn_error_now])

            # plot apical error
            U_H = neuron_data[neuron_data.sender.isin(hidden_id)]

            apical_error = np.linalg.norm(np.stack(U_H.groupby("time_ms")["V_m.a_lat"].apply(np.array).values), axis=1)
            apical_errors.extend(apical_error)
            ax1.plot(utils.rolling_avg(apical_errors, size=150))

            apical_error_now = np.mean(apical_error)
            ax1_2 = ax1.secondary_yaxis("right")
            ax1_2.set_yticks([apical_error_now])

            # Synaptic weights
            # notice that all weights are scaled up again to ensure that derived metrics are comparible between simulations
            WHY = net.get_weight_array(out, hidden) * net.weight_scale
            WHI = net.get_weight_array(interneurons, hidden) * net.weight_scale
            WYH = net.get_weight_array(hidden, out) * net.weight_scale
            WIH = net.get_weight_array(hidden, interneurons) * net.weight_scale

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
