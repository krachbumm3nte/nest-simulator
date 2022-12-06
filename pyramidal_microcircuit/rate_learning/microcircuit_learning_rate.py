import time
import nest
import matplotlib.pyplot as plt
import numpy as np
from params_rate import *
from scipy.ndimage import uniform_filter1d
import pandas as pd
from network_rate import Network
from pympler.tracker import SummaryTracker
from sklearn.metrics import mean_squared_error as mse

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

T = []
w_pi_errors = []
w_ip_errors = []


net = Network()

in_ = net.pyr_pops[0]
in_id = in_.get("global_id")

hidden = net.pyr_pops[1]
hidden_id = hidden.get("global_id")

out = net.pyr_pops[2]
out_id = out.get("global_id")

interneurons = net.intn_pops[0]
intn_id = interneurons.get("global_id")

print(in_id)
print(hidden_id)
print(out_id)
print(intn_id)

"""conn = nest.GetConnections(net.pyr_pops[1][0], net.intn_pops[0][0])
conn.eta = 0.00001
conn.weight = -conn.weight"""

# tracker = SummaryTracker()
np.seterr('raise')
print("setup complete, running simulations...")

for run in range(n_runs):
    inputs = 2 * np.random.rand(dims[0]) -1 
    # input_index = 0
    net.set_input(inputs)

    start = time.time()
    nest.Simulate(SIM_TIME)
    t = time.time() - start
    T.append(t)
    # f = pd.DataFrame.from_dict(net.sr_out.events)["senders"].value_counts()
    # out_activity = [f[i] if (i in f) else 0 for i in out_id]
    # f = pd.DataFrame.from_dict(net.sr_hidden.events)["senders"].value_counts()
    # pyr_activity = [f[i] if (i in f) else 0 for i in hidden_id]
    # f = pd.DataFrame.from_dict(net.sr_intn.events)["senders"].value_counts()
    # intn_activity = [f[i] if (i in f) else 0 for i in intn_id]
    # f = pd.DataFrame.from_dict(net.sr_in.events)["senders"].value_counts()
    # in_activity = [f[i] if (i in f) else 0 for i in in_id]
    # net.sr_out.n_events = 0
    # net.sr_hidden.n_events = 0
    # net.sr_intn.n_events = 0
    # net.sr_in.n_events = 0
    # target_spikes = len(np.where(out_activity == target_id)[0])
    # spike_ratio = target_spikes/total_out_spikes
    # accuracy.append(spike_ratio)
    # if run % 100 == 0:
    # tracker.print_diff()

    # if run > 40:
    #     nest.GetConnections(net.pyr_pops[1], net.intn_pops[0]).set({"eta":  0.0004})
    #     nest.GetConnections(net.intn_pops[0], net.pyr_pops[1]).set({"eta":  0.00001})

    if run % 50 == 0:
        plot_start = time.time()

        time_progressed = run * SIM_TIME

        fig, axes = plt.subplots(3, 2, constrained_layout=True)

        [[ax0, ax1], [ax2, ax3], [ax4, ax5]] = axes
        # for ax in axes.flatten():
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.rcParams['savefig.dpi'] = 300

        # plot somatic voltages of hidden interneurons and output pyramidal neurons
        v_intn = pd.DataFrame.from_dict(net.mm_intn.events).sort_values("times")
        v_intn = v_intn.groupby("senders")

        v_pyr = pd.DataFrame.from_dict(net.mm_out.events).sort_values("times")
        v_pyr = v_pyr.groupby("senders")

        for intn, pyr in zip(intn_id, out_id):
            data_intn = v_intn.get_group(intn)
            ax0.plot(data_intn['times'].array, uniform_filter1d(
                data_intn["V_m.s"].array, size=250), "--", color=cmap_2(intn % dims[2]), alpha=0.5)

            data_pyr = v_pyr.get_group(pyr)
            ax0.plot(data_pyr['times'].array, uniform_filter1d(
                data_pyr['V_m.s'].array, size=250), color=cmap_2(pyr % dims[2]))

        # plot interneuron error
        errors = []
        for intn, pyr in zip(intn_id, out_id):
            int_v = v_intn.get_group(intn)["V_m.s"].array
            pyr_v = v_pyr.get_group(pyr)["V_m.s"].array

            error = np.abs(int_v-pyr_v)
            errors.append(error)

        for i, error in enumerate(errors):
            ax1.plot(uniform_filter1d(error, size=150), color=cmap_2(i % dims[2]), alpha=0.3)
        mean_error = uniform_filter1d(np.mean(errors, axis=0), size=200)
        ax1.plot(mean_error, color="black")
        intn_error_now = np.mean(mean_error[-20:])
        ax1_2 = ax1.secondary_yaxis("right")
        ax1_2.set_yticks([intn_error_now])

        # plot apical voltage
        events = pd.DataFrame.from_dict(net.mm_hidden.events).sort_values("times")
        apical_voltage = events.groupby("senders")

        for id in hidden_id:
            v = apical_voltage.get_group(id)
            ax2.plot(v['times'], uniform_filter1d(v["V_m.a_lat"], size=150), label=id)

        # plot apical error
        apical_err = uniform_filter1d(events.abs().groupby("times")["V_m.a_lat"].mean(), size=150)
        ax3.plot(apical_err, label="apical error")
        ax3_2 = ax3.secondary_yaxis("right")
        apical_err_now = np.mean(apical_err[-20:])
        ax3_2.set_yticks([apical_err_now])

        # plot weight error
        WHY = pd.DataFrame.from_dict(nest.GetConnections(source=out, target=hidden).get())
        WHI = pd.DataFrame.from_dict(nest.GetConnections(source=interneurons, target=hidden).get())
        WYH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=out).get())
        WIH = pd.DataFrame.from_dict(nest.GetConnections(source=hidden, target=interneurons).get())

        a = WHY.sort_values(["target", "source"])
        b = WHI.sort_values(["target", "source"])
        w_ip_error = mse(a["weight"], -b["weight"]) * 10
        print(f"int_pyr error: {w_ip_error}")
        w_ip_errors.append((time_progressed, w_ip_error))
        ax3.plot(*zip(*w_ip_errors), label=f"FB error x10: {w_ip_error:.2f}")

        a = WYH.sort_values(["source", "target"])
        b = WIH.sort_values(["source", "target"])
        w_pi_error = mse(a["weight"], b["weight"]) * 10
        print(f"pyr_int error: {w_pi_error}")
        w_pi_errors.append((time_progressed, w_pi_error))
        ax3.plot(*zip(*w_pi_errors), label=f"FF error x10: {w_pi_error:.2f}")

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

        # ax4.set_ylim(wmin, wmax)
        # ax5.set_ylim(wmin, wmax)
        ax1.set_ylim(bottom=0)
        ax3.set_ylim(bottom=0)

        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, prop={'size': 5})
        # ax2.legend(loc='upper right', ncol=dims[1], prop={'size': 5})
        plt.savefig(f"{run}_weights.png")
        plt.close()
        plot_duration = time.time() - plot_start
        # print(
        #     f"{run}: {np.mean(T[-50:]):.2f}s. spikes: in: {in_activity}, pyr:{pyr_activity}, out:{out_activity}, intn:{intn_activity}")
        print(
            f"{run}: {np.mean(T[-50:]):.2f}s. plot time:{plot_duration:.2f}s apical error: {apical_err_now:.2f}, intn error: {intn_error_now:.2f}")
