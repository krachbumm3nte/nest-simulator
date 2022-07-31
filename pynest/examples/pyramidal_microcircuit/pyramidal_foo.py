import time
import nest
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from params import *

nest.resolution = resolution
nest.set_verbosity("M_ERROR")
nest.SetKernelStatus({"local_num_threads": 1})

dims = [10, 20, 5]

stim = nest.Create("poisson_generator", dims[0])

l0 = nest.Create("parrot_neuron", dims[0])
l1_p = nest.Create(pyr_model, dims[1]//2, pyr_params)
l1_i = nest.Create(intn_model, dims[1]//2, intn_params)
l2 = nest.Create(intn_model, dims[2], intn_params)
out = nest.Create("spike_recorder", dims[2])
# nudge = nest.Create("dc_generator", dims[2])
mm = nest.Create('multimeter', 1, {'record_from': ["V_m.a", "V_m.b"]})


nest.Connect(stim, l0, conn_spec="one_to_one")
nest.Connect(l0, l1_p, syn_spec=syn_ff_pyr_pyr)
nest.Connect(l1_p, l1_i, syn_spec=syn_laminar_pyr_intn)
nest.Connect(l1_p, l2, syn_spec=syn_ff_pyr_pyr)
nest.Connect(l1_i, l1_p, syn_spec=syn_laminar_intn_pyr)
nest.Connect(l2, l1_p, syn_spec=syn_fb_pyr_pyr)
nest.Connect(mm, l1_p)

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
a1.plot(errors_b, 'o', label="u_b")
a2.plot(T, label="t")
a2.set_ylim(0, max(T))
plt.legend()
plt.show()
