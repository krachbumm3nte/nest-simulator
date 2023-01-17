import nest
import matplotlib.pyplot as plt
import numpy as np
from params import *
import pandas as pd
from networks.network_nest import NestNetwork
from sklearn.metrics import mean_squared_error as mse
from time import time
import utils as utils
import os

imgdir, datadir = utils.setup_simulation()
utils.setup_nest(delta_t, sim_params["threads"], sim_params["record_interval"], datadir)
setup_models(True)


weight_scale = 250
syn_params["hi"]["eta"] /= weight_scale**2 * weight_scale * 0.5
syn_params["ih"]["eta"] /= weight_scale**2 * weight_scale * 200

neuron_params["gamma"] = weight_scale
neuron_params["pyr"]["gamma"] = weight_scale
neuron_params["intn"]["gamma"] = weight_scale
neuron_params["input"]["gamma"] = weight_scale
syn_params["wmin_init"] = -1/weight_scale
syn_params["wmax_init"] = 1/weight_scale

g_lk_dnd = delta_t  # TODO: this is neat, but why is it correct?
neuron_params["pyr"]["basal"]["g_L"] = g_lk_dnd
neuron_params["pyr"]["apical_lat"]["g_L"] = g_lk_dnd
neuron_params["intn"]["basal"]["g_L"] = g_lk_dnd

net = NestNetwork(sim_params, neuron_params, syn_params)

dims = sim_params["dims"]

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

plot_interval = 250

T = []
w_pi_errors = []
w_ip_errors = []


in_ = net.pyr_pops[0]
in_id = in_.get("global_id")

hidden = net.pyr_pops[1]
hidden_id = hidden.get("global_id")

out = net.pyr_pops[2]
out_id = out.get("global_id")

interneurons = net.intn_pops[0]
intn_id = interneurons.get("global_id")
print("setup complete, running simulations...")

nest.PrintNodes()

for run in range(sim_params["n_runs"] + 1):
    inputs = np.random.rand(dims[0])
    # input_index = 0
    net.set_input(inputs)

    start = time()
    net.simulate(sim_params["SIM_TIME"])
    t = time() - start
    T.append(t)

    if run % plot_interval == 0:
        print(f"plotting run {run}")
        start = time()

        time_progressed = run * sim_params["SIM_TIME"]

        fig, axes = plt.subplots(3, 2, constrained_layout=True)

        [[ax0, ax1], [ax2, ax3], [ax4, ax5]] = axes
        # for ax in axes.flatten():
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.rcParams['savefig.dpi'] = 300

        # plot somatic voltages of hidden interneurons and output pyramidal neurons
        U_I = utils.read_data(net.mm_i.global_id, datadir).sort_values("time_ms")
        U_I = U_I.groupby("sender")

        U_Y = utils.read_data(net.mm_y.global_id, datadir).sort_values("time_ms")
        U_Y = U_Y.groupby("sender")

        intn_errors = []

        for intn, pyr in zip(intn_id, out_id):
            data_intn = U_I.get_group(intn)
            col = cmap_2(intn % dims[2])
            ax0.plot(data_intn['time_ms'].array, utils.rolling_avg(
                data_intn["V_m.s"].array, size=250), "--", color=col, alpha=0.5)

            data_pyr = U_Y.get_group(pyr)
            ax0.plot(data_pyr['time_ms'].array, utils.rolling_avg(data_pyr['V_m.s'].array, size=250), color=col)

            int_v = U_I.get_group(intn)["V_m.s"].array
            pyr_v = U_Y.get_group(pyr)["V_m.s"].array

            error = np.square(int_v-pyr_v)
            intn_errors.append(error)
            # plot interneuron error
            ax1.plot(utils.rolling_avg(error, size=150), color=col, alpha=0.35, linewidth=0.7)

        mean_error = utils.rolling_avg(np.mean(intn_errors, axis=0), size=250)
        ax1.plot(mean_error, color="black")

        intn_error_now = np.mean(mean_error[-20:])
        ax1_2 = ax1.secondary_yaxis("right")
        ax1_2.set_yticks([intn_error_now])

        # plot apical voltage
        U_H = utils.read_data(net.mm_h.global_id, datadir).sort_values("time_ms")
        apical_voltage = U_H.groupby("sender")

        for id in hidden_id:
            v = apical_voltage.get_group(id)
            ax2.plot(v['time_ms'], utils.rolling_avg(v["V_m.a_lat"], size=150), label=id)

        # plot apical error
        apical_err = utils.rolling_avg(U_H.abs().groupby("time_ms")["V_m.a_lat"].mean(), size=150)
        ax3.plot(apical_err, label="apical error")
        ax3_2 = ax3.secondary_yaxis("right")
        apical_err_now = np.mean(apical_err[-20:])
        ax3_2.set_yticks([apical_err_now])

        # plot weight error
        WHY = pd.DataFrame.from_dict(nest.GetConnections(source=out, target=hidden).get())
        WHI = pd.DataFrame.from_dict(nest.GetConnections(source=interneurons, target=hidden).get())
        WYH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=out).get())
        WIH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=interneurons).get())

        for w_df in [WHY, WHI, WIH, WYH]:
            w_df.weight *= weight_scale

        a = WHY.sort_values(["target", "source"])
        b = WHI.sort_values(["target", "source"])
        w_ip_error = mse(a["weight"], -b["weight"])
        w_ip_errors.append(w_ip_error)
        error_scale = np.arange(0, (run+1), plot_interval) * sim_params["SIM_TIME"]/sim_params["record_interval"]
        ax3.plot(error_scale, w_ip_errors, label=f"FB error: {w_ip_error:.3f}")

        a = WYH.sort_values(["source", "target"])
        b = WIH.sort_values(["source", "target"])
        w_pi_error = mse(a["weight"], b["weight"])
        w_pi_errors.append(w_pi_error)
        ax3.plot(error_scale, w_pi_errors, label=f"FF error: {w_pi_error:.3f}")

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

        ax0.set_title("intn(--) and pyr(-) somatic voltages")
        ax1.set_title("interneuron - pyramidal error")
        ax2.set_title("apical compartment voltages")
        # ax3.set_title("apical error")
        ax4.set_title("Feedback weights")
        ax5.set_title("Feedforward weights")

        ax1.set_ylim(bottom=0)
        ax3.set_ylim(bottom=0)
        # ax4.set_ylim(-1, 1)
        # ax5.set_ylim(-1, 1)

        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, prop={'size': 5})

        # ax2.legend(loc='upper right', ncol=dims[1], prop={'size': 5})
        plt.savefig(os.path.join(imgdir, f"{run}.png"))
        plt.close()

        plot_duration = time() - start
        print(f"mean simulation time: {np.mean(T[-50:]):.2f}s. plot time:{plot_duration:.2f}s. \
apical error: {apical_err_now:.2f}.")
        print(f"ff error: {w_pi_error:.3f}, fb error: {w_ip_error:.3f}, interneuron error: {intn_error_now:.2f}\n")

    elif run % 50 == 0:
        print(f"run {run} completed.")
