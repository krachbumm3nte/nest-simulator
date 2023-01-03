import nest
import matplotlib.pyplot as plt
from params_rate_test import *

# This script shows that the current connection which transmits somatic voltage to a single target neuron
# neuron model behaves as intended and causes appropriate changes in the neuron dynamics.

pyr_y = nest.Create(pyr_model, 1, pyr_params)
mm_y = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_y, pyr_y)
pyr_y.set({"soma": {"g": tau_input}, "basal": {"g": 0}, "apical_lat": {"g": 0}, "tau_m": tau_input})

intn = nest.Create(pyr_model, 1, pyr_params)
mm_i = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_i, intn)
intn.set({"basal": {"g": 0}, "apical_lat": {"g": 0}})

pyr_id = intn.get("global_id")
pyr_y.target = pyr_id


U_i = 0
U_y = 0

sim_times = [50 for i in range(3)]
stim_amps = [2, -2, 0]
SIM_TIME = sum(sim_times)


UI = []
UY = []


for T, amp in zip(sim_times, stim_amps):
    pyr_y.set({"soma": {"I_e": amp*tau_input}})
    nest.Simulate(T)
    for i in range(int(T/delta_t)):
        delta_u_y = -U_y + amp
        U_y = U_y + (delta_t/tau_x) * delta_u_y

        delta_u_i = -(g_l + g_d + g_a) * U_i + g_si * U_y
        U_i = U_i + delta_t * delta_u_i
        UI.append(U_i)
        UY.append(U_y)


fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
som_in = mm_y.get("events")['V_m.s']
ax0.plot(mm_y.get("events")["times"]/delta_t, som_in, label="NEST")
ax0.plot(UY, label="analytical")

som_out = mm_i.get("events")['V_m.s']
ax1.plot(mm_i.get("events")["times"]/delta_t, som_out, label="NEST")

ax1.plot(UI, label="analytical")

ax0.set_title("input neuron")
ax1.set_title("output neuron")

ax0.legend()
ax1.legend()
plt.show()
