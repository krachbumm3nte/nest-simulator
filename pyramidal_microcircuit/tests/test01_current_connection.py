import nest
import matplotlib.pyplot as plt
from params_rate_test import *

# This script shows that the current connection which transmits somatic voltage to a single target neuron
# neuron model behaves as intended and causes appropriate changes in the neuron dynamics.

pyr_model_rate = "pp_cond_exp_mc_pyr"
pyr_model_rate = "rate_neuron_pyr"

in_nrn = nest.Create(pyr_model_rate, 1, input_params)
mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_in, in_nrn)

trgt_nrn = nest.Create(pyr_model_rate, 1, pyr_params)
mm_trgt = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_trgt, trgt_nrn)
trgt_nrn.set({"basal": {"g": 0}, "apical_lat": {"g": 0}})

trgt_id = trgt_nrn.get("global_id")
in_nrn.target = trgt_id


U_i = 0
U_y = 0

sim_times = [50 for i in range(3)]
stim_amps = [2, -2, 0]
SIM_TIME = sum(sim_times)


UI = []
UY = []


for T, amp in zip(sim_times, stim_amps):
    in_nrn.set({"soma": {"I_e": amp*tau_input}})
    nest.Simulate(T)
    for i in range(int(T/delta_t)):
        delta_u_y = -U_y + amp
        U_y = U_y + (delta_t/tau_x) * delta_u_y

        delta_u_i = -(g_l + g_d + g_a) * U_i + g_si * U_y
        U_i = U_i + delta_t * delta_u_i
        UI.append(U_i)
        UY.append(U_y)


fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
som_in = mm_in.get("events")['V_m.s']
ax0.plot(mm_in.get("events")["times"]/delta_t, som_in, label="NEST")
ax0.plot(UY, label="analytical")

som_out = mm_trgt.get("events")['V_m.s']
ax1.plot(mm_trgt.get("events")["times"]/delta_t, som_out, label="NEST")

ax1.plot(UI, label="analytical")

ax0.set_title("input neuron")
ax1.set_title("output neuron")

ax0.legend()
ax1.legend()
plt.show()
