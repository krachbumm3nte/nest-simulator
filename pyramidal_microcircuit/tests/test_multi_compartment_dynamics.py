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
pyr_in = nest.Create(pyr_model, 1, pyr_params)
mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_in, pyr_in)
pyr_in.set({"soma": {"g": tau_input}, "basal": {"g": 0}, "apical_lat": {"g": 0}, "tau_m": tau_input})

pyr_h = nest.Create(pyr_model, 1, pyr_params)
mm_h = nest.Create("multimeter", 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a_lat"]})
nest.Connect(mm_h, pyr_h)

pyr_out = nest.Create(pyr_model, 1, pyr_params)
mm_out = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_out, pyr_out)


w0 = 1
syn_yh.update({"weight": w0})
nest.Connect(pyr_in, pyr_h, syn_spec=syn_yh)

w1 = 0.5
syn_hy.update({"weight": w1})
nest.Connect(pyr_out, pyr_h, syn_spec=syn_hy)


U_i = 0
U_y = 0
U_h = 0
U_bh = 0
U_ah = 0

sim_times = [250 for i in range(3)]
stim_amps = [2, 1, 0.5]
SIM_TIME = sum(sim_times)


UI = []
UY = []
UH = []
UBH = []
UAH = []

for T, amp in zip(sim_times, stim_amps):
    pyr_in.set({"soma": {"I_e": amp*tau_input}})  # input current needs to be attenuated with input time constant
    # target current needs to be attenuated with target current conductance, as doing so in the neuron model would fuck up the line above...
    pyr_out.set({"soma": {"I_e": phi_inverse(amp) * g_s}})
    print(phi_inverse(amp))
    nest.Simulate(T)

    for i in range(int(T/delta_t)):

        delta_u_i = -U_i + amp
        delta_u_y = -(g_l + g_d + g_a) * U_y + g_s * phi_inverse(amp)
        delta_u_h = -(g_l + g_d + g_a) * U_h + U_bh * g_d + U_ah * g_a

        U_bh = phi(U_i) * w0
        U_ah = phi(U_y) * w1
        U_i = U_i + (delta_t/tau_x) * delta_u_i

        U_y = U_y + delta_t * delta_u_y

        U_h = U_h + delta_t * delta_u_h

        UI.append(U_i)
        UH.append(U_h)
        UY.append(U_y)
        UBH.append(U_bh)
        UAH.append(U_ah)


fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, sharey=True)

ax0.plot(mm_in.get("events")["times"]/delta_t, mm_in.get("events")['V_m.s'], label="NEST computed")
ax0.plot(UI, label="analytical")

ax1.plot(mm_h.get("events")['times']/delta_t, mm_h.get("events")['V_m.a_lat'], label="NEST computed")
ax1.plot(UAH, label="analytical")

ax2.plot(mm_h.get("events")['times']/delta_t, mm_h.get("events")['V_m.b'], label="NEST computed")
ax2.plot(UBH, label="analytical")

ax3.plot(mm_h.get("events")["times"]/delta_t, mm_h.get("events")['V_m.s'], label="NEST computed")
ax3.plot(UH, label="analytical")

ax4.plot(mm_out.get("events")["times"]/delta_t, mm_out.get("events")['V_m.s'], label="NEST computed")
ax4.plot(UY, label="analytical")

ax0.set_title("input neuron voltage")
ax1.set_title("output neuron apical voltage")
ax2.set_title("output neuron basal voltage")
ax3.set_title("output neuron somatic voltage")

ax0.legend()
ax1.legend()
ax2.legend()
plt.show()
