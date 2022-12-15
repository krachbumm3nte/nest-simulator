import nest
import matplotlib.pyplot as plt
import sys
from params_rate_test import *
import numpy as np

"""
This script shows that the neuron model handles membrane voltage updates exactly as described in the analytical case.
The neuron in the hidden layer (pyr_h) recieves synaptic input to both apical and basal dendrites, as well as a direct
somatic current connection from one of the input neurons.
"""


def phi(x):
    return 1 / (1.0 + np.exp(-x))


# TODO: find out which combination of paramters gets the right result here, and DOCUMENT YOUR SHIT!
tau_x = 3
input_filter = 1/tau_x

pyr_params['basal']['g'] = 0
pyr_params['apical_lat']['g'] = 0

pyr_h = nest.Create(pyr_model, 1, pyr_params)
mm_h = nest.Create("multimeter", 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a_lat"]})
nest.Connect(mm_h, pyr_h)
pyr_h.set({'soma': {'g_L': g_lk_som}, 'apical_lat': {'g': g_a}, 'basal': {'g': g_b_pyr, 'g_L': 1}})


pyr_i = nest.Create(pyr_model, 1, pyr_params)
mm_i = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_i, pyr_i)
pyr_i.set({'soma': {'g_L': input_filter}, 'tau_m': input_filter})


pyr_y = nest.Create(pyr_model, 1, pyr_params)
mm_y = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_y, pyr_y)
pyr_y.set({'soma': {'g_L': input_filter}, 'tau_m': 1/tau_x})


w0 = 1
syn_ff_pyr_pyr.update({"weight": w0, "eta": 0.0})
nest.Connect(pyr_i, pyr_h, syn_spec=syn_ff_pyr_pyr)

w1 = 0.5
syn_ff_pyr_pyr.update({"weight": w1})
nest.Connect(pyr_y, pyr_h, syn_spec=syn_ff_pyr_pyr)


U_i = 0
U_y = 0
U_h = 0
U_bh = 0
U_ah = 0

sim_times = [250 for i in range(3)]
stim_amps = [2, -2, 0]
SIM_TIME = sum(sim_times)


UI = []
UY = []
UH = []
UBH = []
UAH = []

for T, amp in zip(sim_times, stim_amps):
    pyr_i.set({"soma": {"I_e": amp*input_filter}})
    pyr_y.set({"soma": {"I_e": -amp*input_filter}})
    nest.Simulate(T)

    for i in range(int(T/resolution)):

        delta_u_i = -U_i + amp
        U_i = U_i + (resolution/tau_x) * delta_u_i

        delta_u_y = -U_y - amp
        U_y = U_y + (resolution/tau_x) * delta_u_y

        U_bh = phi(U_i) * w0 + phi(U_y) * w1

        # hidden neuron voltage is computed from two synaptic connection and a direct current connection
        delta_u_h = -g_lk_som * U_h + U_bh * g_b_pyr + U_ah * g_a
        U_h = U_h + (resolution) * delta_u_h

        UI.append(U_i)
        UH.append(U_h)
        UY.append(U_y)
        UBH.append(U_bh)
        UAH.append(U_ah)


fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey=True)

ax0.plot(mm_i.get("events")["times"]/resolution, mm_i.get("events")['V_m.s'], label="NEST computed")
ax0.plot(UI, label="analytical")

ax1.plot(mm_h.get("events")['times']/resolution, mm_y.get("events")['V_m.s'], label="NEST computed")
ax1.plot(UY, label="analytical")

ax2.plot(mm_h.get("events")['times']/resolution, mm_h.get("events")['V_m.b'], label="NEST computed")
ax2.plot(UBH, label="analytical")

ax3.plot(mm_h.get("events")["times"]/resolution, mm_h.get("events")['V_m.s'], label="NEST computed")
ax3.plot(UH, label="analytical")

ax0.set_title("i1 voltage")
ax1.set_title("i2 voltage")
ax2.set_title("output neuron basal voltage")
ax3.set_title("output neuron somatic voltage")

ax0.legend()
ax1.legend()
ax2.legend()
plt.show()
