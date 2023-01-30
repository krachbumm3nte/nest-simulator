import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")

from params import *  # nopep8
from utils import *  # nopep8

"""
This script shows, that by scaling the output of the neuronal transfer function inversely proportional
to synaptic weights, spiking neuron behaviour converges to that of rate neurons
"""

weight = np.random.random() * 2 - 1  # just to show that it really is indpendent from the weight

tau_x = neuron_params["tau_x"]


detla_t = 0.1
g_lk_dnd = delta_t

U_x = 0
delta_u_x = 0
V_bh = 0

U_x_record = []

V_bh_record = []

sim_time = 500
amps = np.random.random(5)
start, stop, N = 0, 10, 11
spike_multipliers = np.logspace(start, stop, N)
exponents = np.linspace(start, stop, N)

V_dend = np.zeros((len(amps) * sim_time, len(spike_multipliers)))
total_spikes = np.zeros(len(spike_multipliers), dtype=int)

gamma = 3


def phi(x):
    return gamma * np.log(1 + np.exp(beta * (x - theta)))

for a, amp in enumerate(amps):
    for t_it in range(sim_time):
        # rate based solution
        # delta_u_x = -U_x + amp
        # U_x = U_x + (delta_t/tau_x) * delta_u_x
        U_x = amp
        V_bh = weight * phi(U_x)

        # spiking solution
        t = a * sim_time + t_it
        for i, factor in enumerate(spike_multipliers):
            n_spikes = np.random.poisson(delta_t * factor * phi(U_x))
            V_dend[t, i] = V_dend[t - 1, i] + n_spikes * weight / factor - (g_lk_dnd * V_dend[t - 1, i])
            total_spikes[i] += n_spikes
        U_x_record.append(U_x)
        V_bh_record.append(V_bh)

fig, axes = plt.subplots(ncols=4, nrows=2)

axes = axes.flatten()

axes[0].plot(U_x_record)
axes[0].set_title("Input neuron somatic voltage")
for i in range(5):
    mult = i * 2
    axes[i+1].plot(V_dend[:, mult])
    axes[i+1].plot(V_bh_record)
    axes[i+1].set_title(f"factor: 10^{exponents[mult]:.1f}\n spikes: {total_spikes[mult]} ({total_spikes[mult]/(sim_time * len(amps)/1000)}Hz)")
# axes[1].plot(rolling_avg(V_bh_record_spiking, size=2500))

axes[6].plot(spike_multipliers, total_spikes)
axes[6].set_xlabel("factor")
axes[6].set_ylabel("n_spikes")
axes[7].plot(spike_multipliers, spike_multipliers/total_spikes)
axes[7].set_ylabel("factor/n_spikes")


print(f"factor/total_spikes = {spike_multipliers[-1]/total_spikes[-1]:.8f}")

plt.tight_layout()
plt.legend()
plt.show()
