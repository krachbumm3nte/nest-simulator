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


dims = [8, 8, 8]
noise = True
SIM_TIME = 1000
stim_rate = 2500

L = len(dims)
pyr_pops = []
intn_pops = []


stim = nest.Create("poisson_generator", dims[0])
l0 = nest.Create("parrot_neuron", dims[0])
pyr_pops.append(l0)
if noise:
    gauss = nest.Create("noise_generator", 1, {"mean": 0., "std": 0.005})

for lr in range(1, L):
    pyr_l = nest.Create(pyr_model, dims[lr], pyr_params)

    nest.Connect(pyr_pops[-1], pyr_l, syn_spec=syn_ff_pyr_pyr)
    print(lr, len(pyr_l))


    if lr < L -1:
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


out = nest.Create("spike_recorder", dims[-1])
nudge = nest.Create("dc_generator", dims[-1])
nest.Connect(nudge, pyr_pops[-1], "one_to_one", syn_spec={'receptor_type': pyr_comps['soma_curr']})
mm_pyr_l = nest.Create('multimeter', 1, {'record_from': ["V_m.a_lat", "V_m.a_td", "V_m.b", "V_m.s"]})
mm_intn = nest.Create('multimeter', 1, {'record_from': ["V_m.s"]})
mm_pyr_m = nest.Create('multimeter', 1, {'record_from': ["V_m.s"]})

for pop in pyr_pops:
    print(pop)
nest.Connect(mm_pyr_l, pyr_pops[1])
nest.Connect(mm_pyr_m, pyr_pops[2])
nest.Connect(mm_intn, intn_pops[0])

sr_intn = nest.Create("spike_recorder", 1)
sr_pyr = nest.Create("spike_recorder", 1)

nest.Connect(intn_pops[0], sr_intn)
nest.Connect(pyr_pops[2], sr_pyr)




nest.Connect(stim, pyr_pops[0], conn_spec="one_to_one")

record_neuron = pyr_pops[1][0]
record_id = record_neuron.get("global_id")



#print(nest.GetConnections(record_neuron))
#print(nest.GetConnections(target=record_neuron))

T = []

print("setup complete, running simulations...")
np.seterr('raise')
for run in range(1000):
    for i in range(dims[0]):
        if np.random.random() > 0.8:
            stim[i].rate = stim_rate
            #nudge[i].amplitude = 2
        else:
            stim[i].rate = 0
            #nudge[i].amplitude = -2


    start = time.time()
    nest.Simulate(SIM_TIME)
    t = time.time() - start
    T.append(t)
    
    
    
    if run % 25 == 0 and run > 0:



        # uniform_filter1d(np.abs(events["V_m.a_td"][indices]), size=1250)

        # print(f"{run}: {np.mean(T[-50:]):.2f}, td: {td:.6f}, lat: {lat:.6f}, apical potential: {abs(td+lat):.4f}")
        print(f"{run}: {np.mean(T[-50:]):.2f}")

        weight_df = pd.DataFrame.from_dict(wr.events)

        weights_from = regroup_df(weight_df[weight_df.senders == record_id], 'targets')
        weights_to = regroup_df(weight_df[weight_df.targets == record_id], 'senders')

        events_senders = regroup_records(mm_pyr_l.get("events"), "senders")


        fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2)
        fig.set_dpi(fig.get_dpi() * 4)


        times_pyr = sr_pyr.events['times']
        times_int = sr_intn.events['times']

        foo = np.array([times_pyr, times_int])

        ax0.hist(foo, run, density=True, histtype='bar')

        # for k, v in weights_to.items():
            # ax0.plot(v['times'], v['weights'], "r", label=f"from {k}")

        #for k, v in weights_from.items():
            #ax0.plot(v['times'], v['weights'], "g", label=f"from {k}")

        for k, v in events_senders.items():
            ax1.plot(v['times'], uniform_filter1d(np.abs(v["V_m.a_td"]), size=1250), '.', label=f"V_m.a ({k})")
        #f = np.array(data_arrays[0])
        #plt.plot(f[:,0], f[:,1])
        

        for k, v in regroup_records(mm_pyr_m.events, "senders").items():
            ax2.plot(v['times'], uniform_filter1d(v['V_m.s'], size=1250), '.')
        
        for k, v in regroup_records(mm_intn.events, "senders").items():
            ax3.plot(v['times'], uniform_filter1d(v['V_m.s'], size=1250), '.')


        ax0.set_title("pyramidal neuron weights")
        ax1.set_title("apical compartment voltage")
        #ax0.legend()
        # ax1.legend()
        #ax0.set_ylim(-1, 1)
        #ax0.hlines(0, 0, run * SIM_TIME)

        plt.savefig(f"{run}_weights.png")
        plt.close()
        # plt.show()
