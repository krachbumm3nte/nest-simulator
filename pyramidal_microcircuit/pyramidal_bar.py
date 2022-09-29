import time
import nest
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from params import *

nest.resolution = resolution
nest.set_verbosity("M_ERROR")
nest.SetKernelStatus({"local_num_threads": 1})

dims = [3, 5, 3]
L = len(dims)

pyr_pops = []
intn_pops = []

stim = nest.Create("poisson_generator", dims[0])
l0 = nest.Create("parrot_neuron", dims[0])
pyr_pops.append(l0)

for l in range(1, L-1):
    pyr_l = nest.Create(pyr_model, dims[l], pyr_params)
    intn_l = nest.Create(intn_model, dims[l+1], intn_params)

    nest.Connect(pyr_pops[-1], pyr_l, syn_spec=syn_ff_pyr_pyr)
    nest.Connect(pyr_l, intn_l, syn_spec=syn_laminar_pyr_intn)
    nest.Connect(intn_l, pyr_l, syn_spec=syn_laminar_intn_pyr)
    if l > 1:
        nest.Connect(pyr_l, pyr_pops[-1], syn_spec=syn_fb_pyr_pyr)
        nest.Connect()

    pyr_pops.append(pyr_l)
    intn_pops.append(intn_l)

out_neurons = nest.Create(pyr_model, dims[-1], pyr_params)
nest.Connect(out_neurons, pyr_pops[-1], syn_spec=syn_fb_pyr_pyr)
nest.Connect(pyr_pops[-1], out_neurons, syn_spec=syn_ff_pyr_pyr)

out = nest.Create("spike_recorder", dims[2])
# nudge = nest.Create("dc_generator", dims[2])
mm = nest.Create('multimeter', 1, {'record_from': ["V_m.a", "V_m.b"]})


nest.Connect(stim, l0, conn_spec="one_to_one")
nest.Connect(mm, pyr_pops[1])

errors_a = []
errors_b = []

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
    e = np.mean(np.abs(mm.get("events")["V_m.a"]))
    if run % 50 == 0:
        print(f"{run}: {t:.2f}: {e}")
    errors_a.append(e)
    errors_b.append(np.mean(np.abs(mm.get("events")["V_m.b"])))
    T.append(t)
    mm.n_events = 0

fig, (a1, a2) = plt.subplots(1, 2)

a1.plot(errors_a, 'o', label="u_a")
# a1.plot(errors_b, 'o', label="u_b")
a2.plot(T, label="t")
a2.set_ylim(0, max(T))
plt.legend()
plt.show()
