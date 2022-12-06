import nest
import matplotlib.pyplot as plt
import sys
from params_rate_test import *

# This script shows that the current connection implemented in the pp_cond_exp_mc_pyr
# neuron model behaves as intended and causes appropriate changes in the neuron dynamics.


tau_x = 3
input_filter = 1/tau_x

pyr_params['basal']['g'] = 0
pyr_params['apical_lat']['g'] = 0
pyr_params['tau_m'] = 1


pyr_in = nest.Create(pyr_model, 1, pyr_params)
mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_in, pyr_in)
pyr_in.set({'soma': {'g_L': input_filter}})

pyr_out = nest.Create(pyr_model, 1, pyr_params)
mm_out = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_out, pyr_out)

pyr_id = pyr_out.get("global_id")
pyr_in.target = pyr_id


U_i = 0
U_y = 0

sim_times = [50 for i in range(3)]
stim_amps = [2, -2, 0]
SIM_TIME = sum(sim_times)


UI = []
UY = []


for T, amp in zip(sim_times, stim_amps):
    pyr_in.set({"soma": {"I_e": amp*input_filter}})
    nest.Simulate(T)

    for i in range(int(T/resolution)):
        delta_u_y = -U_y + amp
        U_y = U_y + (resolution/tau_x) * delta_u_y

        delta_u_i = -(g_lk_som) * U_i + lam * U_y
        U_i = U_i + resolution * delta_u_i
        UI.append(U_i)
        UY.append(U_y)


fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
som_in = mm_in.get("events")['V_m.s']
ax0.plot(som_in, label="soma_0")

som_out = mm_out.get("events")['V_m.s']
ax0.plot(som_out, label="soma_1")

ax1.plot(UY, label="input")
ax1.plot(UI, label="connected")

a = (min(UI), max(UI))
b = (min(UY), max(UY))

ax0.hlines(a, 0, SIM_TIME, color="b")
ax1.hlines(a, 0, SIM_TIME/resolution, color="b")

ax0.hlines(b, 0, SIM_TIME, color="r")
ax1.hlines(b, 0, SIM_TIME/resolution, color="r")

ax0.set_title("NEST simulation")
ax1.set_title("analytical computation")

ax0.legend()
ax1.legend()
plt.show()
