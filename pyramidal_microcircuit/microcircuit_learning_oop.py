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


SIM_TIME = 100
n_runs = 500

# dims = [4, 4, 4]
dims = [2, 3, 3]

cmap_1 = plt.cm.get_cmap('hsv', 5)
cmap_2 = plt.cm.get_cmap('hsv', 5)

noise = True
noise_std = 0.1

stim_amp = 0.5

target_amp = 10


T = []

accuracy = []

net = Network(dims, True, noise_std=noise_std, init_self_pred=True)


pyr_1 = net.pyr_pops[1]
pyr_1_id = pyr_1.get("global_id")

pyr_2 = net.pyr_pops[2]
pyr_2_id = pyr_2.get("global_id")

int_0 = net.intn_pops[0]
int_0_id = int_0.get("global_id")

print(pyr_1_id, pyr_2_id, int_0_id)
print(pyr_2.get(["global_id", "target"]))

print(nest.GetConnections(source=pyr_2, target=pyr_1))
print(nest.GetConnections(source=int_0, target=pyr_1))
print(nest.GetConnections(target=pyr_2, source=pyr_1))
print(nest.GetConnections(target=int_0, source=pyr_1))


print("setup complete, running simulations...")
np.seterr('raise')
for run in range(n_runs):
    input_index = int(np.random.random() * dims[0])

    net.set_input([input_index])

    # target_id = pyr_pops[-1][input_index].get("global_id")

    start = time.time()
    nest.Simulate(SIM_TIME)
    t = time.time() - start
    T.append(t)
    out_activity = np.count_nonzero(net.sr_pyr.events["senders"] == pyr_1_id[0])
    net.sr_out.n_events = 0
    total_out_spikes = out_activity
    # target_spikes = len(np.where(out_activity == target_id)[0])
    # spike_ratio = target_spikes/total_out_spikes
    # accuracy.append(spike_ratio)

    if run % 5 == 0 and run > 0:

        # uniform_filter1d(np.abs(events["V_m.a_td"][indices]), size=1250)

        # print(f"{run}: {np.mean(T[-50:]):.2f}, td: {td:.6f}, lat: {lat:.6f}, apical potential: {abs(td+lat):.4f}")
        print(f"{run}: {np.mean(T[-50:]):.2f}. num spikes: {total_out_spikes}")

        weight_df = pd.DataFrame.from_dict(wr.events)

        weights_from = regroup_df(weight_df[weight_df.senders == pyr_1_id], 'targets')
        weights_to = regroup_df(weight_df[weight_df.targets == pyr_1_id], 'senders')

        events_senders = regroup_records(net.mm_pyr_l.get("events"), "senders")

        fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2)
        fig.set_dpi(fig.get_dpi() * 4)

        times_pyr = net.sr_pyr.events['times']
        times_int = net.sr_intn.events['times']
        net.sr_pyr.n_events = 0
        net.sr_intn.n_events = 0

        foo = times_pyr, times_int

        v_pyr_l = regroup_records(net.mm_pyr_l.events, 'senders')
        v_pyr_m = regroup_records(net.mm_pyr_m.events, 'senders')
        v_int = regroup_records(net.mm_intn.events, 'senders')

        td_weights = np.array(nest.GetConnections(pyr_2, pyr_1).get("weight"))
        spikes_pyr = regroup_records(net.sr_pyr.events, "senders")

        """for k, v in v_int.items():
            ax0.plot(v['times'], uniform_filter1d(v['V_m.s'], size=3000), 'g')

        for k, v in v_pyr_l.items():
            ax0.plot(v['times'], uniform_filter1d(v['V_m.s'], size=3000), 'r')
        """
        foo = [v['times'] for (k, v) in spikes_pyr.items()]
        ax0.set_title("pyramidal spike rates")
        ax0.hist(foo, run//10 + 1, density=False)

        for k, v in events_senders.items():
            ax1.plot(v['times'], uniform_filter1d(np.abs(v["V_m.a_lat"]), size=900), label=f"V_m.a ({k})")
        # ax1.set_ylim(0, 2)
        ax1.set_title("apical compartment voltage")

        w_pp_21 = pd.DataFrame.from_dict(nest.GetConnections(source=pyr_2, target=pyr_1).get())
        w_ip_11 = pd.DataFrame.from_dict(nest.GetConnections(source=int_0, target=pyr_1).get())
        w_pp_12 = pd.DataFrame.from_dict(nest.GetConnections(source=pyr_1, target=pyr_2).get())
        w_pi_11 = pd.DataFrame.from_dict(nest.GetConnections(source=pyr_1, target=int_0).get())

        for idx, row in w_pp_21.iterrows():
            t = row["target"]
            col = cmap_2(row["source"] % dims[2])
            ax2.plot(row["target"], row["weight"], ".", color=col, label=f"to {t}")

        for idx, row in w_ip_11.iterrows():
            t = row["target"]
            ax2.plot(row["target"], row["weight"], ".", color=cmap_2(row["source"] % dims[2]), label=f"from {t}")

        for idx, row in w_pp_12.iterrows():
            t = row["target"]
            ax3.plot(row["source"], row["weight"], ".", color=cmap_2(row["target"] % dims[2]), label=f"to {t}")

        for idx, row in w_pi_11.iterrows():
            t = row["target"]
            ax3.plot(row["source"], -row["weight"], ".", color=cmap_2(row["target"] % dims[2]), label=f"from {t}")

        """
        ax3.hlines(td_weights*1.1, 0, run*SIM_TIME, 'blue')
        for k, v in weights_to.items():
            ax3.plot(v['times'], v['weights'], "g", label=f"from {k}")

        for k, v in weights_from.items():
            ax3.plot(v['times'], v['weights'], "r", label=f"to {k}")
"""
        # ax3.hlines(0, 0, run*SIM_TIME, 'black')
        ax3.set_title("intn_pyr weights")

        plt.savefig(f"{run}_weights.png")
        plt.close()
