import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
import nest
from params import *
from utils import *

def phi(x):
    return 1 / (1 + np.exp(-x))


def p_spike(x):
    return -1 * (np.exp(-phi(x) * 0.1) - 1)


voltage = 15.2

weight = .5


phi_max = 0.9
rate_slope = 0.9
beta = 3.
theta = 1


def phi_old(x):
    return phi_max / (1.0 + rate_slope * np.exp(beta * (theta - x)))

# phi function from sacramento (2018) Fig S1
def phi_sac(x):
    return np.log(1 + np.exp(x))


def p_spike_2(x):
    return -1 * (np.exp(-phi_sac(x) * 0.1) - 1)


gen = nest.Create("dc_generator")
par = nest.Create("pp_cond_exp_mc_pyr", pyr_params)

pyr = nest.Create("pp_cond_exp_mc_pyr", pyr_params)

nest.Connect(gen, par, syn_spec={"receptor_type": pyr_comps["soma_curr"]})
syn_ff_pyr_pyr["weight"] = weight
nest.Connect(par, pyr, syn_spec=syn_ff_pyr_pyr)

sr = nest.Create("spike_recorder")

nest.Connect(par, sr)
nest.Connect(pyr, sr)

gen.amplitude=voltage

nest.Simulate(25)

spikes = regroup_records(sr.events, "senders")
print(spikes)
c = ["r", "g"]
for i, (k,v) in enumerate(spikes.items()):
    plt.vlines(v["times"], 0, 1, c[i], label=k)

#plt.plot(x, p_spike(x), label="p_spike_1")
#plt.plot(x, p_spike_2(x), label="p_spike_2")

plt.legend()
plt.show()
