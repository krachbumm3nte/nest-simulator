import nest
import matplotlib.pyplot as plt
import sys
from params_rate_test import *
import numpy as np

"""
This script shows that the neuron model handles a single basal input exactly like the analytical
solution if parameters are set correctly.
"""


def phi(x):
    return 1 / (1.0 + np.exp(-x))

wr = nest.Create('weight_recorder')
nest.CopyModel("pyr_synapse_rate", "p_syn", {"weight_recorder": wr})
syn_ff_pyr_pyr["synapse_model"] = "p_syn"


# plasticity in the simulator is slightly faster than in the analytical model! this is due to information
# in the synapse being delayed. For this proof of concept I have multiplied the weight changes in the 
# analytical solution with this magic number.
magic_plasticity_number = 1.25
tau_x = 3
input_filter = 1/tau_x

pyr_params['basal']['g'] = 0
pyr_params['apical_lat']['g'] = 0
# Important realization: for voltages to match exactly, tau_m needs to be equal to simualtion resolution!

pyr_in = nest.Create(pyr_model, 1, pyr_params)
mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_in, pyr_in)
# In the input neuron, tau_m needs to be increased to match the low-pass filtering of input changes.
pyr_in.set({'soma': {'g_L': input_filter}, 'tau_m': input_filter})

pyr_h = nest.Create(pyr_model, 1, pyr_params)
mm_h = nest.Create("multimeter", 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a_lat"]})
nest.Connect(mm_h, pyr_h)
pyr_h.set({'soma': {'g_L': g_lk_som}, 'apical_lat': {'g': 0}, 'basal': {'g': g_b_pyr, 'g_L': 1}})

eta = 0.03
w0 = 0.5
syn_ff_pyr_pyr.update({"weight": w0, "eta": eta})
nest.Connect(pyr_in, pyr_h, syn_spec=syn_ff_pyr_pyr)


U_i = 0
U_h = 0
U_bh = 0

sim_times = [120 for i in range(3)]
stim_amps = [1, -1, 0]

# sim_times = [1 for i in range(1)]
# stim_amps = [1]


SIM_TIME = sum(sim_times)


UI = []
UH = []
UBH = []
W0 = []
tilde_w = 0

for T, amp in zip(sim_times, stim_amps):
    pyr_in.set({"soma": {"I_e": amp*input_filter}})
    nest.Simulate(T)

    for i in range(int(T/resolution)):

        delta_u_i = -U_i + amp
        U_i = U_i + (resolution/tau_x) * delta_u_i
        
        vw_star = phi((g_b_pyr * U_bh)/(g_lk_som))
        dend_error = (phi(U_h) - vw_star)
        delta_tilde_w = -tilde_w + dend_error * phi(U_i)
        tilde_w = tilde_w + (resolution * delta_tilde_w) / tau_delta
        w0 = w0 + eta * resolution * tilde_w * magic_plasticity_number


        print(phi(U_i), w0, tilde_w, delta_tilde_w, dend_error, vw_star)


        U_bh = phi(U_i) * w0
        # print("s: ", U_i, U_bh, U_h)
        delta_u_h = -g_lk_som * U_h + U_bh * g_b_pyr
        U_h = U_h + (resolution) * delta_u_h



        UI.append(U_i)
        UH.append(U_h)
        UBH.append(U_bh)
        W0.append(w0)


fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey=False)

ax0.plot(mm_in.get("events")["times"]/resolution, mm_in.get("events")['V_m.s'], label="NEST computed")
ax0.plot(UI, label="analytical")

ax1.plot(mm_h.get("events")["times"]/resolution , mm_h.get("events")['V_m.s'], label="NEST computed")
ax1.plot(UH, label="analytical")

ax2.plot(mm_h.get("events")['times']/resolution , mm_h.get("events")['V_m.b'], label="NEST computed")
ax2.plot(UBH, label="analytical")

ax3.plot(wr.get("events")["times"]/resolution , wr.get("events")["weights"], label="NEST computed")
ax3.plot(W0, label="analytical")

ax0.set_title("input neuron voltage")
ax1.set_title("output neuron somatic voltage")
ax2.set_title("output neuron basal voltage")

ax0.legend()
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()
