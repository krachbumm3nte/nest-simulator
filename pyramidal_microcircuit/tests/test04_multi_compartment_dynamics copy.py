import nest
import matplotlib.pyplot as plt
from params_rate_test import *

"""
This script shows that the neuron model handles membrane voltage updates exactly as described in the analytical case.
The neuron in the hidden layer (pyr_h) recieves synaptic input to both apical and basal dendrites.
"""
pyr_x = nest.Create(pyr_model, 1, pyr_params)
mm_x = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_x, pyr_x)
pyr_x.set({"soma": {"g": tau_input}, "tau_m": tau_input})

pyr_h = nest.Create(pyr_model, 1, pyr_params)
mm_h = nest.Create("multimeter", 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a_lat"]})
nest.Connect(mm_h, pyr_h)

pyr_y = nest.Create(pyr_model, 1, pyr_params)
mm_y = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_y, pyr_y)


w0 = 1
syn_yh.update({"weight": w0})
nest.Connect(pyr_x, pyr_h, syn_spec=syn_yh)

w1 = 0.5
syn_hy.update({"weight": w1})
nest.Connect(pyr_y, pyr_h, syn_spec=syn_hy)


U_x = 0
U_y = 0
U_h = 0
V_bh = 0
U_ah = 0

sim_times = [250 for i in range(3)]
stim_amps = [2, 1, 0.5]
target_amps = [0.1, 0.5, 1]

SIM_TIME = sum(sim_times)


UX = [0]
UH = [0]
UY = [0]
UBH = [0]
UAH = [0]

for T, stim, target in zip(sim_times, stim_amps, target_amps):
    pyr_x.set({"soma": {"I_e": stim*tau_input}})  # input current needs to be attenuated with input time constant
    # target current needs to be attenuated with target current conductance, as doing so in the neuron model would
    # fuck up the line above...
    pyr_y.set({"soma": {"I_e": phi_inverse(target) * g_s}})
    nest.Simulate(T)

    for i in range(int(T/delta_t)):

        delta_u_i = -U_x + stim
        delta_u_h = -(g_l + g_d + g_a) * U_h + V_bh * g_d + U_ah * g_a
        delta_u_y = -(g_l + g_d + g_a) * U_y + g_s * phi_inverse(target)

        V_bh = phi(U_x) * w0
        U_ah = phi(U_y) * w1
        U_x = U_x + (delta_t/tau_x) * delta_u_i

        U_y = U_y + delta_t * delta_u_y

        U_h = U_h + delta_t * delta_u_h

        UX.append(U_x)
        UH.append(U_h)
        UY.append(U_y)
        UBH.append(V_bh)
        UAH.append(U_ah)


fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, sharex=True)

ax0.plot(mm_x.get("events")["times"]/delta_t, mm_x.get("events")['V_m.s'], label="NEST computed")
ax0.plot(UX, label="analytical")

ax1.plot(mm_h.get("events")['times']/delta_t, mm_h.get("events")['V_m.b'], label="NEST computed")
ax1.plot(UBH, label="analytical")

ax2.plot(mm_h.get("events")["times"]/delta_t, mm_h.get("events")['V_m.s'], label="NEST computed")
ax2.plot(UH, label="analytical")

ax3.plot(mm_h.get("events")['times']/delta_t, mm_h.get("events")['V_m.a_lat'], label="NEST computed")
ax3.plot(UAH, label="analytical")

ax4.plot(mm_y.get("events")["times"]/delta_t, mm_y.get("events")['V_m.s'], label="NEST computed")
ax4.plot(UY, label="analytical")

ax0.set_title("input neuron voltage")
ax1.set_title("hidden neuron basal voltage")
ax2.set_title("hidden neuron somatic voltage")
ax3.set_title("hidden neuron apical voltage")
ax4.set_title("output neuron somatic voltage")

ax0.legend()
plt.show()