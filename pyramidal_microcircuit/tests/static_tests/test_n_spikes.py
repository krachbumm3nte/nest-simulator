import sys
import nest
import matplotlib.pyplot as plt
import os
from copy import deepcopy
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")
import utils  # nopep8
from params import *  # nopep8


root, imgdir, datadir = utils.setup_simulation(
        "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/tests/runs")
utils.setup_nest(sim_params, datadir)
utils.setup_models(True, neuron_params, sim_params, syn_params)

weight_scale = 2500
n1 = nest.Create(neuron_params["model"], 1, neuron_params["pyr"])
n2 = nest.Create(neuron_params["model"], 1, neuron_params["pyr"])
sr1 = nest.Create("spike_recorder")
sr2 = nest.Create("spike_recorder")
nest.Connect(n1, sr1)
nest.Connect(n2, sr2)

n1.gamma = weight_scale * neuron_params["pyr"]["gamma"]
n2.gamma = neuron_params["pyr"]["gamma"]
n1.set({"soma": {"I_e": 3}})
n2.set({"soma": {"I_e": 3}})




nest.Simulate(1000000)

n_events_1, n_events_2 = sr1.n_events, sr2.n_events
print(n_events_1, n_events_2, n_events_1/n_events_2)