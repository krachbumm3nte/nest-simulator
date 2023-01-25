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

imgdir, datadir = utils.setup_simulation()
utils.setup_nest(sim_params, datadir)

spiking = False

setup_models(spiking)


if spiking:
    weight_scale = 100
    syn_params["hi"]["eta"] /= weight_scale**3 * 4
    syn_params["ih"]["eta"] /= weight_scale**3 * 330
    syn_params["hx"]["eta"] /= weight_scale**3 * 330
    syn_params["yh"]["eta"] /= weight_scale**3 * 330

    neuron_params["gamma"] = weight_scale
    neuron_params["pyr"]["gamma"] = weight_scale
    neuron_params["intn"]["gamma"] = weight_scale
    neuron_params["input"]["gamma"] = weight_scale
    syn_params["wmin_init"] = -1/weight_scale
    syn_params["wmax_init"] = 1/weight_scale

    sim_params["noise"] = False

net = NestNetwork(sim_params, neuron_params, syn_params)

dims = sim_params["dims"]

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

plot_interval = 10

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
print("setup complete, running simulations...")

# dump simulation parameters to a .json file
with open(os.path.join(os.path.dirname(imgdir), "params.json"), "w") as f:
    json.dump({"simulation": sim_params, "neurons": neuron_params, "synapses": syn_params}, f)

batchsize = 100

try:
    for run in range(sim_params["n_runs"] + 1):
        inputs = np.random.rand(dims[0])
        # input_index = 0
        start = time()
        # net.train(inputs, sim_params["SIM_TIME"])
        net.train_batches(sim_params["SIM_TIME"], batchsize)
        t = (time() - start) / batchsize
        T.append(t)

        if run % plot_interval == 0:
            print(f"plotting run {run}")
            start = time()
            fig, axes = plt.subplots(4, 2, constrained_layout=True)
            [[ax0, ax1], [ax2, ax3], [ax4, ax5], [ax6, ax7]] = axes
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

            abs_voltage = np.mean(abs_voltage)
            mean_error = utils.rolling_avg(np.mean(intn_errors, axis=0), size=2)
            ax0.plot(mean_error, color="black")

            intn_error_now = np.mean(mean_error[-20:])
            ax0_2 = ax0.secondary_yaxis("right")
            ax0_2.set_yticks([intn_error_now])

            # plot apical error
            U_H = neuron_data[neuron_data.sender.isin(hidden_id)]

            apical_error = U_H.abs().groupby("time_ms")["V_m.a_lat"].mean()
            apical_errors.extend(apical_error.values)
            ax1.plot(apical_errors)

            apical_error_now = np.mean(apical_error.values)
            ax1_2 = ax1.secondary_yaxis("right")
            ax1_2.set_yticks([apical_error_now])

            # Synaptic weights
            WHY = pd.DataFrame.from_dict(nest.GetConnections(source=out, target=hidden).get())
            WHI = pd.DataFrame.from_dict(nest.GetConnections(source=interneurons, target=hidden).get())
            WYH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=out).get())
            WIH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=interneurons).get())

            # plot weight error
            WHY = WHY.sort_values(["target", "source"])
            WHI = WHI.sort_values(["target", "source"])
            fb_error = mse(WHY["weight"], -WHI["weight"])
            fb_errors.append(fb_error)
            error_scale = np.arange(0, (run+1), plot_interval) * sim_params["SIM_TIME"]/sim_params["record_interval"]
            ax2.plot(error_scale, fb_errors, label=f"FB error: {fb_error:.3f}")

            WYH = WYH.sort_values(["source", "target"])
            WIH = WIH.sort_values(["source", "target"])
            ff_error = mse(WYH["weight"], WIH["weight"])
            ff_errors.append(ff_error)
            ax3.plot(error_scale, ff_errors, label=f"FF error: {ff_error:.3f}")

            # plot weights
            for idx, row in WHY.iterrows():
                t = row["target"]
                col = cmap_2(row["source"] % dims[2])
                ax4.plot(row["target"], row["weight"], ".", color=col, label=f"to {t}")

            for idx, row in WHI.iterrows():
                t = row["target"]
                ax4.plot(row["target"], -row["weight"], "x", color=cmap_2(row["source"] % dims[2]), label=f"from {t}")

            for idx, row in WYH.iterrows():
                t = row["target"]
                ax5.plot(row["source"], row["weight"], ".", color=cmap_2(row["target"] % dims[2]), label=f"to {t}")

            for idx, row in WIH.iterrows():
                t = row["target"]
                ax5.plot(row["source"], row["weight"], "x", color=cmap_2(row["target"] % dims[2]), label=f"from {t}")

            ax6.plot(net.output_loss)

            ax0.set_title("interneuron - pyramidal error")
            ax1.set_title("apical error")
            ax2.set_title("Feedback error")
            ax3.set_title("Feedforward error")
            ax4.set_title("Feedback weights")
            ax5.set_title("Feedforward weights")
            ax6.set_title("Output loss")

            ax0.set_ylim(bottom=0)
            ax1.set_ylim(bottom=0)
            ax2.set_ylim(bottom=0)
            ax3.set_ylim(bottom=0)

            plt.savefig(os.path.join(imgdir, f"{run*batchsize}.png"))
            plt.close()

            plot_duration = time() - start
            print(f"mean simulation time: {np.mean(T[-10:]):.3f}s. plot time:{plot_duration:.2f}s. \
    apical error: {apical_error_now:.2f}.")
            print(f"ff error: {ff_error:.4f}, fb error: {fb_error:.4f}, interneuron error: {intn_error_now:.4f}, absolute somatic voltage: {abs_voltage:.3f}\n")

        elif run % 100 == 0:
            print(f"run {run} completed.")
except KeyboardInterrupt:
    print("KeyboardInterrupt received - storing synaptic weights...")
finally:
    utils.store_synaptic_weights(net, os.path.dirname(datadir))
    print("weights stored to disk, exiting.")