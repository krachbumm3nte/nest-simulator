import json
import os

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.params import Params

import nest

plot_utils.setup_plt()

simulation_dir = "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/results/bars_le_full_plast_deep_rnest"
weight_loc = os.path.join(simulation_dir, "data/weights_1000.json")
p = Params(os.path.join(simulation_dir, "params.json"))
p.weight_scale = 500
p.t_pres = 5
p.out_lag = 0
p.test_delay = 0
p.test_time = 5
p.spiking = True
p.latent_equilibrium = True
p.init_self_pred = False
p.dims = [9, 30, 10, 3]
utils.setup_nest(p)

net = NestNetwork(p)

sr_pyr = nest.Create("spike_recorder")
sr_intn = nest.Create("spike_recorder")

n_samples = 5


def reset_recorders():
    sr_pyr.n_events = 0
    sr_intn.n_events = 0


pyr_spikes_test = []
intn_spikes_test = []

pyr_spikes_train = []
intn_spikes_train = []


for l in net.layers[:-1]:
    nest.Connect(l.pyr, sr_pyr)
    nest.Connect(l.intn, sr_intn)
nest.Connect(net.layers[-1].pyr, sr_pyr)

print("training with random weights")

x, y = net.get_training_data(n_samples)

for i in range(3):
    if i == 1:
        net.set_selfpredicting_weights()
        print("training with selfpred weights")
    elif i == 2:
        with open(weight_loc) as f:
            wgts = json.load(f)
        net.set_all_weights(wgts)
        print("training with final weights")

    net.test_batch(x, y)
    pyr_spikes_test.append(sr_pyr.n_events)
    intn_spikes_test.append(sr_intn.n_events)
    reset_recorders()

    net.train_batch(x, y)
    pyr_spikes_train.append(sr_pyr.n_events)
    intn_spikes_train.append(sr_intn.n_events)
    reset_recorders()


width = 0.3
ind = np.arange(3)
fig, ax = plt.subplots()
ax.bar(ind, intn_spikes_test, width=width, color="b")
ax.bar(ind, pyr_spikes_test, width=width, color="r", bottom=intn_spikes_test)

ax.bar(ind+width, intn_spikes_train, width=width, color="b", hatch='//')
ax.bar(ind+width, pyr_spikes_train, width=width, color="r", bottom=intn_spikes_test, hatch='//')


ax.set_ylabel(r"$N_{spikes}$")
ax.set_xticks(ind+0.5*width, ["Random weights", "Selfpredicting weights", "After training"])
ax.legend(labels=["Interneurons", "Pyramidal neurons"])
plt.savefig("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/data/activity_unpredicted_stimulus.png")
