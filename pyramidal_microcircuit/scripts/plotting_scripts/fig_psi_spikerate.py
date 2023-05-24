# -*- coding: utf-8 -*-
#
# fig_psi_spikerate.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.plot_utils as plot_utils
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.params import Params

import nest

plot_utils.setup_plt()

p = Params()
p.t_pres = 10
p.out_lag = 0
p.store_errors = True
p.init_self_pred = False
p.eta = {
    "ip": [
        0.0,
        0.0
    ],
    "pi": [
        0.0,
        0.0
    ],
    "up": [
        0.0,
        0.0
    ],
    "down": [
        0,
        0
    ]
}

# p.gamma = 0.1
# p.beta = 1
# p.theta = 3

psi_list = [0.001, 0.01, 0.1, 1, 10]


n_min = []
n_max = []
n_mean = []

for psi in psi_list:
    print(f"psi = {psi}")
    p.psi = psi

    utils.setup_nest(p)

    net = NestNetwork(p)

    sr = nest.Create("spike_recorder")

    all_neurons = nest.GetNodes({"model": p.neuron_model})

    nest.Connect(all_neurons, sr)

    net.train_samples = 10
    net.train_epoch()

    events = pd.DataFrame.from_dict(sr.events)
    grouped_events = events.groupby("senders")

    Hz = [10*len(i) for i in grouped_events.groups.values()]

    n_min.append([psi, min(Hz)])
    n_max.append([psi, max(Hz)])
    n_mean.append([psi, np.mean(Hz)])
    nest.ResetKernel()


plt.plot(*zip(*n_min), label="min")
plt.plot(*zip(*n_max), label="max")
plt.plot(*zip(*n_mean), label="mean")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\psi$")
plt.ylabel("Hz")
plt.legend()
plt.show()
