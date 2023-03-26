import argparse
import json
import os
import sys
import warnings
from datetime import timedelta
from time import time

import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params
from src.plot_utils import plot_training_progress
import nest

params = Params()
params.record_interval = 0.1
params.threads = 1
weight_scales = np.logspace(0, 1, 100, endpoint=True)


acc = []

utils.setup_nest(params)
params.setup_nest_configs()
pyr = nest.Create(params.neuron_model, 1, params.pyr_params)
intn = nest.Create(params.neuron_model, 1, params.intn_params)
input_nrn = nest.Create(params.neuron_model, 1, params.input_params)

sr = nest.Create("spike_recorder")
nest.Connect(pyr, sr)
nest.Connect(intn, sr)
nest.Connect(input_nrn, sr)

pyr_hz = []
intn_hz = []
input_nrn_hz = []
sim_time = 1000


def get_hz_from_sr(spike_recorder, neuron, sim_time):
    n_spikes = len(np.where(spike_recorder.events["senders"] == neuron.global_id)[0])
    return (1000 * n_spikes) / sim_time


for scale in weight_scales:
    sr.n_events = 0
    pyr.gamma = params.gamma * scale
    intn.gamma = params.gamma * scale
    input_nrn.gamma = scale

    pyr.set({"soma": {"I_e": 0.5 * params.g_som}})
    intn.set({"soma": {"I_e": 0.5 * params.g_som}})
    input_nrn.set({"soma": {"I_e": 0.5 / params.tau_x}})

    nest.Simulate(sim_time)

    pyr_hz.append([scale, get_hz_from_sr(sr, pyr, sim_time)])
    intn_hz.append([scale, get_hz_from_sr(sr, intn, sim_time)])
    input_nrn_hz.append([scale, get_hz_from_sr(sr, input_nrn, sim_time)])

plt.xscale("log")
plt.plot(*zip(*sorted(pyr_hz)), label="pyr")
plt.plot(*zip(*sorted(intn_hz)), label="intn")
plt.plot(*zip(*sorted(input_nrn_hz)), label="input")
plt.show()
