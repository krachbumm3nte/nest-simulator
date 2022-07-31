import time
import nest
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from params import *

nest.resolution = resolution
nest.set_verbosity("M_ERROR")
nest.SetKernelStatus({"local_num_threads": 2})

dims = [30, 20, 10]

stim = nest.Create("poisson_generator", dims[0])

l0 = nest.Create("parrot_neuron", dims[0])

l1_p = nest.Create(pyr_model, dims[1]//2, pyr_params)
l1_i = nest.Create(intn_model, dims[1]//2, intn_params)

l2 = nest.Create(intn_model, dims[2], intn_params)

out = nest.Create("spike_recorder", dims[2])

#nudge = nest.Create("dc_generator", dims[2])

nest.Connect(stim, l0, conn_spec="one_to_one")

nest.Connect(l0, l1_p, syn_spec=syn_ff_pyr_pyr)

nest.Connect(l1_p, l1_i, syn_spec=syn_laminar_pyr_intn)

nest.Connect(l1_i, l1_p, syn_spec=syn_laminar_intn_pyr)

nest.Connect(l1_p, l2, syn_spec=syn_ff_pyr_pyr)

nest.Connect(l2, l1_p, syn_spec=syn_fb_pyr_pyr)

#nest.Connect(nudge, l2, conn_spec="one_to_one", syn_spec={"receptor_type": intn_comps['soma_curr']})


mm = nest.Create('multimeter', 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a"]})
nest.Connect(mm, l1_p)

errors = []

for run in range(1000):
    start = time.time()
    for i in range(dims[0]):
        if np.random.random() > 0.8:
            stim[i].rate = 400
        else:
            stim[i].rate = 0

    nest.Simulate(300)
    e = np.mean(np.abs(mm.get("events")["V_m.a"]))
    t = time.time() - start
    print(f"{t:.2f}: {e}")
    errors.append(e)
    mm.n_events = 0

plt.plot(errors)
