import time
import nest
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from params import *
from scipy.ndimage import uniform_filter1d

import pandas as pd


def regroup_records(records, group_key):
    records = pd.DataFrame.from_dict(records)
    return regroup_df(records, group_key)


def regroup_df(df, group_key):
    return dict([(n, x.loc[:, x.columns != group_key]) for n, x in df.groupby(group_key)])


SIM_TIME = 100
n_runs = 500

# dims = [4, 4, 4]
dims = [30, 20, 10]

noise = True
noise_std = 0.1

stim_amp = 1

target_amp = 10


L = len(dims)
pyr_pops = []
intn_pops = []


l0 = nest.Create(pyr_model, dims[0], pyr_params)
pyr_pops.append(l0)
if noise:
    gauss = nest.Create("noise_generator", 1, {"mean": 0., "std": noise_std})

for lr in range(1, L):
    pyr_l = nest.Create(pyr_model, dims[lr], pyr_params)

    nest.Connect(pyr_pops[-1], pyr_l, syn_spec=syn_ff_pyr_pyr)
    print(lr, len(pyr_l))

    if lr < L - 1:
        int_l = nest.Create(intn_model, dims[lr+1], intn_params)
        print(len(int_l))

        if noise:
            nest.Connect(gauss, int_l, syn_spec={"receptor_type": intn_comps["soma_curr"]})

        nest.Connect(pyr_l, int_l, syn_spec=syn_laminar_pyr_intn)
        nest.Connect(int_l, pyr_l, syn_spec=syn_laminar_intn_pyr)

        intn_pops.append(int_l)

    if lr > 1:
        print(f"setting up {len(pyr_l)} feedback connections")
        for i in range(len(pyr_l)):
            id = int_l[i].get("global_id")
            pyr_l[i].target = id

        nest.Connect(pyr_l, pyr_pops[-1], syn_spec=syn_fb_pyr_pyr)

    if noise:
        nest.Connect(gauss, pyr_l, syn_spec={"receptor_type": pyr_comps["soma_curr"]})
    pyr_pops.append(pyr_l)


nudge = nest.Create("dc_generator", dims[-1], {'amplitude': 0})
nest.Connect(nudge, pyr_pops[-1], "one_to_one", syn_spec={'receptor_type': pyr_comps['soma_curr']})

mm_pyr_l = nest.Create('multimeter', 1, {'record_from': ["V_m.a_lat", "V_m.a_td", "V_m.b", "V_m.s"]})
mm_intn = nest.Create('multimeter', 1, {'record_from': ["V_m.s"]})
mm_pyr_m = nest.Create('multimeter', 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_pyr_l, pyr_pops[1])
nest.Connect(mm_pyr_m, pyr_pops[2])
nest.Connect(mm_intn, intn_pops[0])

sr_intn = nest.Create("spike_recorder", 1)
sr_pyr = nest.Create("spike_recorder", 1)
sr_out = nest.Create("spike_recorder", 1)
nest.Connect(intn_pops[0], sr_intn)
nest.Connect(pyr_pops[1], sr_pyr)
nest.Connect(pyr_pops[-1], sr_out)

stim = nest.Create("dc_generator", dims[0])
nest.Connect(stim, pyr_pops[0], conn_spec="one_to_one", syn_spec={"receptor_type": pyr_comps['soma_curr']})

record_neuron = pyr_pops[1][0]
record_id = record_neuron.get("global_id")

T = []

accuracy = []

print("setup complete, running simulations...")
np.seterr('raise')
for run in range(n_runs):
    input_index = int(np.random.random() * dims[0])
    for i in range(dims[0]):
        # if i == input_index:
        if np.random.random() > 0.8:
            stim[i].amplitude = stim_amp
            # nudge[i].amplitude = target_amp
        else:
            stim[i].amplitude = 0
            # nudge[i].amplitude = -target_amp

    # target_id = pyr_pops[-1][input_index].get("global_id")

    start = time.time()
    nest.Simulate(SIM_TIME)
    t = time.time() - start
    T.append(t)
    out_activity = sr_out.events["senders"]
    sr_out.n_events = 0
    total_out_spikes = len(out_activity)
    # target_spikes = len(np.where(out_activity == target_id)[0])
    # spike_ratio = target_spikes/total_out_spikes
    # accuracy.append(spike_ratio)

    if run % 5 == 0 and run > 0:

        # uniform_filter1d(np.abs(events["V_m.a_td"][indices]), size=1250)

        # print(f"{run}: {np.mean(T[-50:]):.2f}, td: {td:.6f}, lat: {lat:.6f}, apical potential: {abs(td+lat):.4f}")
        print(f"{run}: {np.mean(T[-50:]):.2f}. out spikes: {total_out_spikes}")

        weight_df = pd.DataFrame.from_dict(wr.events)

        weights_from = regroup_df(weight_df[weight_df.senders == record_id], 'targets')
        weights_to = regroup_df(weight_df[weight_df.targets == record_id], 'senders')

        events_senders = regroup_records(mm_pyr_l.get("events"), "senders")

        fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2)
        fig.set_dpi(fig.get_dpi() * 4)

        times_pyr = sr_pyr.events['times']
        times_int = sr_intn.events['times']

        foo = times_pyr, times_int

        v_pyr_l = regroup_records(mm_pyr_l.events, 'senders')
        v_pyr_m = regroup_records(mm_pyr_m.events, 'senders')
        v_int = regroup_records(mm_intn.events, 'senders')

        td_weights = np.array(nest.GetConnections(pyr_pops[2], record_neuron).get("weight")) * -1

        spikes_pyr = regroup_records(sr_pyr.events, "senders")

        for k, v in v_int.items():
            ax0.plot(v['times'], uniform_filter1d(v['V_m.s'], size=3000), 'g')

        for k, v in v_pyr_l.items():
            ax0.plot(v['times'], uniform_filter1d(v['V_m.s'], size=3000), 'r')
        ax0.set_title("pyr somatic voltages")

        for k, v in events_senders.items():
            ax1.plot(v['times'], uniform_filter1d(np.abs(v["V_m.a_td"]), size=900), label=f"V_m.a ({k})")
        # ax1.set_ylim(0, 2)
        ax1.set_title("apical compartment voltage")

        foo = [v['times'] for (k,v) in spikes_pyr.items()]
        ax2.hist(foo, run, density=True)
        # ax2.plot(uniform_filter1d(accuracy, size=400))
        # ax2.set_title("accuracy")

        for k, v in weights_to.items():
            ax3.plot(v['times'], v['weights'], "g", label=f"from {k}")

        for k, v in weights_from.items():
            ax3.plot(v['times'], v['weights'], "r", label=f"to {k}")

        ax3.hlines(td_weights, 0, run*SIM_TIME, 'blue')
        ax3.hlines(0, 0, run*SIM_TIME, 'black')
        ax3.set_title("intn_pyr weights")

        plt.savefig(f"{run}_weights.png")
        plt.close()
