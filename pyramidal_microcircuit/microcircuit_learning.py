import time
import nest
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from params import *

nest.resolution = resolution
nest.set_verbosity("M_ERROR")
nest.SetKernelStatus({"local_num_threads": 1})

dims = [4, 6, 4]
L = len(dims)
noise = True


pyr_pops = []
intn_pops = []


stim = nest.Create("poisson_generator", dims[0])
l0 = nest.Create("parrot_neuron", dims[0])
pyr_pops.append(l0)
if noise:
    gauss = nest.Create("noise_generator", 1, {"mean": 0., "std": 0.1})

for lr in range(1, L):
    pyr_l = nest.Create(pyr_model, dims[lr], pyr_params)

    nest.Connect(pyr_pops[-1], pyr_l, syn_spec=syn_ff_pyr_pyr)

    if 1 <= lr < L:
        int_l = nest.Create(intn_model, dims[lr+1], intn_params)

        nest.Connect(pyr_l, int_l, syn_spec=syn_laminar_pyr_intn)

        nest.Connect(int_l, pyr_l, syn_spec=syn_laminar_intn_pyr)

        print("setting up feedback connections")
        # TODO: do we need top-down current inputs or not?
        for i in range(len(pyr_l)):
            id = int_l[i].get("global_id")
            pyr_l[i].target = id

        nest.Connect(pyr_l, pyr_pops[-1], syn_spec=syn_fb_pyr_pyr)

        if noise:
            nest.Connect(gauss, int_l, syn_spec={"receptor_type": intn_comps["soma_curr"]})
        intn_pops.append(int_l)

    if noise:
        nest.Connect(gauss, pyr_l, syn_spec={"receptor_type": pyr_comps["soma_curr"]})
    pyr_pops.append(pyr_l)


out = nest.Create("spike_recorder", dims[-1])
# nudge = nest.Create("dc_generator", dims[2])
mm = nest.Create('multimeter', 1, {'record_from': ["V_m.a_lat", "V_m.a_td", "V_m.b", "V_m.s"]})


nest.Connect(stim, pyr_pops[0], conn_spec="one_to_one")
nest.Connect(mm, intn_pops[0])

print(nest.GetConnections(intn_pops[0][0]))
print(nest.GetConnections(target=intn_pops[0][0]))
pot_td = []
pot_lat = []

T = []

for run in range(1500):
    for i in range(dims[0]):
        if np.random.random() > 0.8:
            stim[i].rate = 400
        else:
            stim[i].rate = 0

    start = time.time()
    nest.Simulate(200)
    t = time.time() - start
    pot_td.append(np.mean(mm.get("events")["V_m.b"]))
    pot_lat.append(np.mean(mm.get("events")["V_m.s"]))

    if run % 50 == 0 and run > 0:
        td = np.mean(pot_td[-50:])
        lat = np.mean(pot_lat[-50:])

        print(f"{run}: {np.mean(T[-50:]):.2f}, td: {td:.6f}, lat: {lat:.6f}, apical potential: {td+lat:.4f}")
    T.append(t)
    mm.n_events = 0
    # TODO: get and print synaptic weights

fig, (a1, a2) = plt.subplots(1, 2)

a1.plot(pot_td, 'o', label="u_a")
# a1.plot(errors_b, 'o', label="u_b")
a2.plot(T, label="t")
a2.set_ylim(0, max(T))
plt.legend()
plt.show()
