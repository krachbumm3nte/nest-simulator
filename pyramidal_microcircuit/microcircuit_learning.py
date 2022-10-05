import time
import nest
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from params import *



dims = [6, 4, 2]
L = len(dims)
noise = True


pyr_pops = []
intn_pops = []


stim = nest.Create("poisson_generator", dims[0])
l0 = nest.Create("parrot_neuron", dims[0])
pyr_pops.append(l0)
if noise:
    gauss = nest.Create("noise_generator", 1, {"mean": 0., "std": 0.2})

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
        # TODO: do we need top-down current inputs or not?
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
mm = nest.Create('multimeter', 1, {'record_from': ["V_m.a_lat", "V_m.a_td", "V_m.b", "V_m.s"]})


nest.Connect(stim, pyr_pops[0], conn_spec="one_to_one")

record_neuron = pyr_pops[1][0]
record_id = record_neuron.get("global_id")
nest.Connect(mm, record_neuron)



print(nest.GetConnections(record_neuron))
print(nest.GetConnections(target=record_neuron))
pot_td = []
pot_lat = []

T = []

for run in range(1500):
    for i in range(dims[0]):
        if np.random.random() > 0.8:
            stim[i].rate = 1000
            #nudge[i].amplitude = 2
        else:
            stim[i].rate = 0
            #nudge[i].amplitude = -2


    start = time.time()
    nest.Simulate(200)
    t = time.time() - start
    T.append(t)
    
    pot_td.append(np.abs(np.mean(mm.get("events")["V_m.a_td"])))
    pot_lat.append(np.abs(np.mean(mm.get("events")["V_m.a_lat"])))
    mm.n_events = 0
    
    if run % 50 == 0 and run > 0:
        td = np.mean(pot_td[-50:])
        lat = np.mean(pot_lat[-50:])

        print(f"{run}: {np.mean(T[-50:]):.2f}, td: {td:.6f}, lat: {lat:.6f}, apical potential: {abs(td+lat):.4f}")

        e = wr.events
        e_arr = np.array([e['senders'], e['targets'], e['receptors'], e['times'], e['weights']])

        out_weights = e_arr[:,np.where(e_arr[0] == record_id)].squeeze(1)
        in_weights = e_arr[:,np.where(e_arr[1] == record_id)].squeeze(1)

        out_ar = {}
        in_ar = {}

        for i, row in enumerate(out_weights.swapaxes(0,1)):
            f = row[1]
            if f in out_ar:
                out_ar[f].append((row[3:]))
            else:
                out_ar[f] = [row[3:]]

        for i, row in enumerate(in_weights.swapaxes(0,1)):
            f = row[0]
            if f in in_ar:
                in_ar[f].append((row[3:]))
            else:
                in_ar[f] = [row[3:]]

        fig, [ax0, ax1] = plt.subplots(1, 2)
        for k, v in in_ar.items():
            f = np.array(v)
            if not np.min(f[:,1]) == np.max(f[:,1]):
                ax0.plot(f[:,0], f[:,1], label=f"from {k}")

        for k, v in out_ar.items():
            f = np.array(v)
            if not np.min(f[:,1]) == np.max(f[:,1]):
                ax0.plot(f[:,0], f[:,1], "g", label=f"to {k}")

        ax1.plot(pot_td, label="top-down apical potential")
        ax1.plot(pot_lat, label="lateral apical potential")
        #f = np.array(data_arrays[0])
        #plt.plot(f[:,0], f[:,1])
        
        ax0.set_title("pyramidal neuron weights")
        ax1.set_title("apical compartment voltage")
        ax0.legend()
        ax0.set_ylim(-1, 1)
        ax1.legend()
        plt.savefig(f"{run}_weights.png")
        # plt.show()






    # TODO: get and print synaptic weights

fig, (a1, a2) = plt.subplots(1, 2)

a1.plot(pot_td, 'o', label="u_a")
# a1.plot(errors_b, 'o', label="u_b")
a2.plot(T, label="t")
a2.set_ylim(0, max(T))
plt.legend()
plt.show()
