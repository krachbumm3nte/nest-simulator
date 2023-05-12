import nest
import argparse
import os
import sys
from time import time
import json
from datetime import timedelta

import numpy as np
import src.utils as utils
from microcircuit_learning import run_simulations
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params
import matplotlib.pyplot as plt

p = Params()
p.network_type = "snest"
p.spiking = True
p.psi = 2
p.setup_nest_configs()
utils.setup_nest(p)


def phi(x):
    return p.gamma * np.log(1 + np.exp(p.beta * (x - p.theta)))


syn = p.syn_static
syn["receptor_type"] = p.compartments['basal']


n_neuron = 784
n_hidden = 200
t_sim = 25

i = 0.7

input_orig = nest.Create(p.neuron_model, n_neuron, p.input_params)
out = nest.Create(p.neuron_model, n_hidden, p.pyr_params)
nest.Connect(input_orig, out, syn_spec=syn)
input_orig.set({"soma": {"I_e": i / p.tau_x}})

print(f"testing original input neurons... ")
t_start = time()
nest.Simulate(t_sim)
t_orig = time() - t_start

print(f"done after {t_orig:.3f}s\n")
nest.ResetKernel()

input_new = nest.Create("poisson_generator", n_neuron)
parrots_new = nest.Create("parrot_neuron", n_neuron)
out = nest.Create(p.neuron_model, n_hidden, p.pyr_params)
nest.Connect(input_new, parrots_new, conn_spec="one_to_one")
nest.Connect(parrots_new, out, syn_spec=syn)
input_new.rate = p.psi * i * 1000

print(f"testing new input neurons... ")

t_start = time()
nest.Simulate(t_sim)
t_new = time() - t_start

print(f"done after {t_new:.3f}s")
