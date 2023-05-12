import nest
import argparse
import os
import sys
import time
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
p.setup_nest_configs()
utils.setup_nest(p)


def phi(x):
    return p.gamma * np.log(1 + np.exp(p.beta * (x - p.theta)))


input_orig = nest.Create(p.neuron_model, 1, p.input_params)

input_new = nest.Create("poisson_generator", 1)

out = nest.Create(p.neuron_model, 2, p.pyr_params)

syn = p.syn_static
syn["receptor_type"] = p.compartments['basal']

nest.Connect(input_orig, out[0], syn_spec=syn)
nest.Connect(input_new, out[1], syn_spec=syn)

mm1 = nest.Create("multimeter", 1, {'record_from': ["V_m.a_lat", "V_m.s", "V_m.b"]})
mm2 = nest.Create("multimeter", 1, {'record_from': ["V_m.a_lat", "V_m.s", "V_m.b"]})

nest.Connect(mm1, out[0])
nest.Connect(mm2, out[1])


i_e = [0.5, 0, 1]

for i in i_e:

    input_orig.set({"soma": {"I_e": i / p.tau_x}})
    input_new.rate = p.psi * i * 1000
    nest.Simulate(50)


fig, [ax0, ax1] = plt.subplots(2, 1)
print(mm1.events)
ax0.plot(mm1.events["times"], mm1.events["V_m.b"], color="blue")
ax0.plot(mm2.events["times"], mm2.events["V_m.b"], color="orange")
ax1.plot(mm1.events["times"], mm1.events["V_m.s"], color="blue")
ax1.plot(mm2.events["times"], mm2.events["V_m.s"], color="orange")

plt.show()