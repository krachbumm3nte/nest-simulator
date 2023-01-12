import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")

from params import *  # nopep8
from utils import *  # nopep8

"""
This script explores different combinations of scaling down synaptic weights and dendritic leakage
condcutance, that enable a spiking neuron to mirror the behaviour of rate neurons as exactly as possible.
"""

amp = 0.5
weight = np.random.random() * 2 - 1  # just to show that it really is indpendent from the weight

tau_x = neuron_params["tau_x"]


weight_scale = 14.3
g_lk_dnd = 0.007

weight_scale = 44.3
g_lk_dnd = 0.0023

weight_scale = 1
g_lk_dnd = 0.095

U_x = 0
delta_u_x = 0
V_bh = 0
V_bh_spiking = 0

U_x_record = []

V_bh_record = []
V_bh_record_spiking = []

n_spikes_total = 0

for amp in [0.5, 1, 0]:
    for i in range(20000):
        delta_u_x = -U_x + amp

        U_x = U_x + (delta_t/tau_x) * delta_u_x

        # rate based solution
        V_bh = weight * U_x

        # spiking solution
        n_spikes = np.random.poisson(delta_t * U_x)
        V_bh_spiking += n_spikes * weight / weight_scale - (g_lk_dnd * V_bh_spiking)

        U_x_record.append(U_x)
        V_bh_record.append(V_bh)
        V_bh_record_spiking.append(V_bh_spiking)
        n_spikes_total += n_spikes

print(f"transmitted {n_spikes_total} spikes.")
fig, axes = plt.subplots(ncols=2)

axes[0].plot(U_x_record)

axes[1].plot(V_bh_record)
axes[1].plot(V_bh_record_spiking, alpha=0.2)
axes[1].plot(rolling_avg(V_bh_record_spiking, size=2500))

plt.legend()
plt.show()
