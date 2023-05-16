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

simulation_dir = "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/results/bars_le_full_plast_snest"
weight_loc = os.path.join(simulation_dir, "weights.json")
p = Params()
p.spiking = True
p.network_type = "snest"
# p.psi = 500
p.t_pres = 25
p.out_lag = 0
p.store_errors = True
p.init_self_pred = False
p.dims = [9, 30, 3]
p.eta = {
        "ip": [
            0.0,
            0.0,
            0.0
        ],
        "pi": [
            0.0,
            0.0,
            0.0
        ],
        "up": [
            0.0,
            0.0,
            0.0
        ],
        "down": [
            0,
            0,
            0
        ]
    }
utils.setup_nest(p)

net = NestNetwork(p)

sr_pyr = nest.Create("spike_recorder")
sr_intn = nest.Create("spike_recorder")

n_samples = 2


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


x, y_batch = net.get_training_data(n_samples)


api_err = []

print("training with random weights")
for i in range(3):
    if i == 1:
        net.set_selfpredicting_weights()
        print("training with selfpred weights")
    elif i == 2:
        with open(weight_loc) as f:
            wgts = json.load(f)
        net.set_all_weights(wgts)
        print("training with final weights")

    net.train_batch(x, y_batch)
    pyr_spikes_train.append(sr_pyr.n_events)
    intn_spikes_train.append(sr_intn.n_events)
    reset_recorders()

    api_err.append(np.mean([i[1] for i in net.apical_error]))
    net.apical_error = []

    # net.test_batch(x, y_batch)
    # pyr_spikes_test.append(sr_pyr.n_events)
    # intn_spikes_test.append(sr_intn.n_events)
    # reset_recorders()


print(api_err)


width = 0.3
gap = 0.005
ind = np.arange(3)
fig, ax = plt.subplots()


# Activity difference between training and testing was irrelevant, as shown by plotting these lines instead.
# ax.bar(ind-gap, intn_spikes_test,  align="edge", width=-width,  color="b")
# ax.bar(ind-gap, pyr_spikes_test, align="edge", width=-width, color="r", bottom=intn_spikes_test)

# ax.bar(ind+gap, intn_spikes_train, align="edge", width=width, color="b", hatch='//')
# ax.bar(ind+gap, pyr_spikes_train, align="edge", width=width, color="r", bottom=intn_spikes_train, hatch='//')

ax.bar(ind, intn_spikes_train, width=width, color="b")
ax.bar(ind, pyr_spikes_train, width=width, bottom=intn_spikes_train, color="r")


ax.set_ylabel(r"$N_{spikes}$")
ax.set_xticks(ind, ["Random weights", "Selfpredicting weights", "After training"])
ax.legend(labels=["Interneurons", "Pyramidal neurons"])
plt.savefig("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/data/fig_unpredicted_stimulus.png")
