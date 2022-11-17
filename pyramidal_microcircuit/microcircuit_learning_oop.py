import sys
import time
import nest
import matplotlib.pyplot as plt
import numpy as np
from params import *
from scipy.ndimage import uniform_filter1d
from utils import *
import pandas as pd
from network import Network
from pympler.tracker import SummaryTracker
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import mean_squared_error as mse

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

T = []
w_pi_errors = []
w_ip_errors = []


net = Network()

pyr_1 = net.pyr_pops[1]
pyr_1_id = pyr_1.get("global_id")

pyr_2 = net.pyr_pops[2]
pyr_2_id = pyr_2.get("global_id")

int_0 = net.intn_pops[0]
int_0_id = int_0.get("global_id")

# print(nest.GetConnections(source=pyr_2, target=pyr_1))
# print(nest.GetConnections(source=int_0, target=pyr_1))
# print(nest.GetConnections(source=pyr_1, target=int_0))
# print(nest.GetConnections(source=pyr_1, target=pyr_2))

# tracker = SummaryTracker()
np.seterr('raise')
print("setup complete, running simulations...")
for run in range(n_runs):
    input_index = int(np.random.random() * dims[0])
    net.set_input([input_index])

    start = time.time()
    nest.Simulate(SIM_TIME)
    t = time.time() - start
    T.append(t)
    out_activity = len(net.sr_out.events["times"])
    pyr_activity = len(net.sr_pyr.events["times"])
    intn_activity = len(net.sr_intn.events["times"])
    in_activity = len(net.sr_in.events["times"])
    net.sr_out.n_events = 0
    net.sr_pyr.n_events = 0
    net.sr_intn.n_events = 0
    net.sr_in.n_events = 0
    # target_spikes = len(np.where(out_activity == target_id)[0])
    # spike_ratio = target_spikes/total_out_spikes
    # accuracy.append(spike_ratio)
    # if run % 100 == 0:
    # tracker.print_diff()

    # if run > 40:
    #     nest.GetConnections(net.pyr_pops[1], net.intn_pops[0]).set({"eta":  0.0004})
    #     nest.GetConnections(net.intn_pops[0], net.pyr_pops[1]).set({"eta":  0.00001})

    if (run == 10 or run % 25 == 0) and run > 0:
        plot_start = time.time()

        time_progressed = run * SIM_TIME

        fig, axes = plt.subplots(3, 2, constrained_layout=True)

        [[ax0, ax1], [ax2, ax3], [ax4, ax5]] = axes
        # for ax in axes.flatten():
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.rcParams['savefig.dpi'] = 300

        # plot somatic voltages of hidden interneurons and output pyramidal neurons
        v_intn = pd.DataFrame.from_dict(net.mm_intn.events).sort_values("times")
        v_intn["senders"] = v_intn["senders"].transform(lambda x: x % dims[2])
        v_intn = v_intn.groupby("senders")

        v_pyr = pd.DataFrame.from_dict(net.mm_pyr_1.events).sort_values("times")
        v_pyr["senders"] = v_pyr["senders"].transform(lambda x: x % dims[2])
        v_pyr = v_pyr.groupby("senders")

        for i in range(dims[2]):
            data_intn = v_intn.get_group(i)
            ax0.plot(data_intn['times'].array, uniform_filter1d(
                data_intn["V_m.s"].array, size=2000), "--", color=cmap_2(i), alpha=0.5)

            data_pyr = v_pyr.get_group(i)
            ax0.plot(data_pyr['times'].array, uniform_filter1d(data_pyr['V_m.s'].array, size=20000), color=cmap_2(i))

        # plot interneuron error
        errors = []
        for i in range(dims[2]):
            int_v = v_intn.get_group(i)["V_m.s"].array
            pyr_v = v_pyr.get_group(i)["V_m.s"].array

            error = np.abs(int_v-pyr_v)
            errors.append(error)

        for id, error in enumerate(errors):
            ax1.plot(uniform_filter1d(error, size=1500), color=cmap_2(id), alpha=0.3)
        mean_error = uniform_filter1d(np.mean(errors, axis=0), size=2000)
        ax1.plot(mean_error, color="black")
        intn_error_now = np.mean(mean_error[-20:])
        ax1_2 = ax1.secondary_yaxis("right")
        ax1_2.set_yticks([intn_error_now])

        # plot apical voltage
        events = pd.DataFrame.from_dict(net.mm_pyr_0.events).sort_values("times")
        events["senders"] = events["senders"].transform(lambda x: x % dims[2])
        apical_voltage = events.groupby("senders")

        for i in range(dims[2]):
            v = apical_voltage.get_group(i)
            ax2.plot(v['times'], uniform_filter1d(v["V_m.a_lat"], size=1500),
                     color=cmap_1(i), label=i)

        # plot apical error
        apical_err = uniform_filter1d(events.abs().groupby("times")["V_m.a_lat"].mean(), size=1500)
        ax3.plot(apical_err, label="apical error")
        ax3_2 = ax3.secondary_yaxis("right")
        apical_err_now = np.mean(apical_err[-20:])
        ax3_2.set_yticks([apical_err_now])

        # plot weight error
        w_pp_21 = pd.DataFrame.from_dict(nest.GetConnections(source=pyr_2, target=pyr_1).get())
        w_ip_11 = pd.DataFrame.from_dict(nest.GetConnections(source=int_0, target=pyr_1).get())
        w_pp_12 = pd.DataFrame.from_dict(nest.GetConnections(source=pyr_1, target=pyr_2).get())
        w_pi_11 = pd.DataFrame.from_dict(nest.GetConnections(source=pyr_1, target=int_0).get())

        a = w_pp_21.sort_values(["target", "source"])
        b = w_ip_11.sort_values(["target", "source"])
        w_ip_error = mse(a["weight"], -b["weight"])
        print(f"int_pyr error: {w_ip_error}")
        w_ip_errors.append((time_progressed, w_ip_error))
        ax3.plot(*zip(*w_ip_errors), label=f"FB error {w_ip_error:.2f}")

        a = w_pp_12.sort_values(["source", "target"])
        b = w_pi_11.sort_values(["source", "target"])
        w_pi_error = mse(a["weight"], b["weight"])
        print(f"pyr_int error: {w_pi_error}")
        w_pi_errors.append((time_progressed, w_pi_error))
        ax3.plot(*zip(*w_pi_errors), label=f"FF error {w_pi_error:.2f}")

        # plot weights
        for idx, row in w_pp_21.iterrows():
            t = row["target"]
            col = cmap_2(row["source"] % dims[2])
            ax4.plot(row["target"], row["weight"], ".", color=col, label=f"to {t}")

        for idx, row in w_ip_11.iterrows():
            t = row["target"]
            ax4.plot(row["target"], -row["weight"], "x", color=cmap_2(row["source"] % dims[2]), label=f"from {t}")

        for idx, row in w_pp_12.iterrows():
            t = row["target"]
            ax5.plot(row["source"], row["weight"], ".", color=cmap_2(row["target"] % dims[2]), label=f"to {t}")

        for idx, row in w_pi_11.iterrows():
            t = row["target"]
            ax5.plot(row["source"], row["weight"], "x", color=cmap_2(row["target"] % dims[2]), label=f"from {t}")

        ax0.set_title("intn(--) and pyr(-) somatic voltages")
        ax1.set_title("interneuron - pyramidal error")
        ax2.set_title("apical compartment voltages")
        ax3.set_title("apical error")
        ax4.set_title("W_PP_FB = -W_IP_FB")
        ax5.set_title("W_PP_FF = W_PI_FF")

        ax4.set_ylim(-1, 1)
        ax5.set_ylim(-1, 1)
        ax1.set_ylim(bottom=0)
        ax3.set_ylim(bottom=0)

        ax3.legend(loc="upper right")

        plt.savefig(f"{run}_weights.png")
        plt.close()
        plot_duration = time.time() - plot_start
        print(f"{run}: {np.mean(T[-50:]):.2f}s. spikes: in: {in_activity}, pyr:{pyr_activity}, out:{out_activity}, intn:{intn_activity}")
        print(
            f"{run}: {np.mean(T[-50:]):.2f}s. plot time:{plot_duration:.2f}s apical error: {apical_err_now:.2f}, intn error: {intn_error_now:.2f}")
