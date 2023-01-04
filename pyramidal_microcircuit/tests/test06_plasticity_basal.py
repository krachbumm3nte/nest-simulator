import nest
import matplotlib.pyplot as plt
from params_rate_test import *

"""
This script shows that the neuron model handles a single basal input exactly like the analytical
solution if parameters are set correctly.
"""
nest.SetDefaults("multimeter", {'interval': 0.1})

pyr_in = nest.Create(pyr_model, 1, pyr_params)
mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_in, pyr_in)
# In the input neuron, tau_m needs to be increased to match the low-pass filtering of input changes.
pyr_in.set({"soma": {"g": tau_input}, "basal": {"g": 0}, "apical_lat": {"g": 0}, "tau_m": tau_input})

pyr_h = nest.Create(pyr_model, 1, pyr_params)
mm_h = nest.Create("multimeter", 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a_lat"]})
nest.Connect(mm_h, pyr_h)
pyr_h.set({'soma': {'g_L': g_l}, 'apical_lat': {'g': g_a}, 'basal': {'g': g_d}})

eta = 0.03
w0 = 0.5
syn_yh.update({"weight": w0, "eta": eta})
nest.Connect(pyr_in, pyr_h, syn_spec=syn_yh)


U_i = 0
U_h = 0
U_bh = 0

sim_times = [100 for i in range(3)]
stim_amps = [1, -1, 0]

SIM_TIME = sum(sim_times)


UI = [0]
UH = [0]
UBH = [0]
W0 = [w0]
tilde_w = 0

for T, amp in zip(sim_times, stim_amps):
    pyr_in.set({"soma": {"I_e": amp*tau_input}})
    nest.Simulate(T)

    for i in range(int(T/delta_t)):

        delta_u_i = -U_i + amp
        delta_u_h = -(g_l + g_d + g_a) * U_h + U_bh * g_d

        vw_star = phi((g_d * U_bh)/(g_l + g_d + g_a))
        dend_error = phi(U_h) - vw_star
        delta_tilde_w = -tilde_w + dend_error * phi(U_i)
        tilde_w = tilde_w + (delta_t * delta_tilde_w) / tau_delta
        w0 = w0 + eta * delta_t * tilde_w
        U_bh = phi(U_i) * w0
        U_i = U_i + (delta_t/tau_x) * delta_u_i
        U_h = U_h + delta_t * delta_u_h

        UI.append(U_i)
        UH.append(U_h)
        UBH.append(U_bh)
        W0.append(w0)


fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey=False)

ax0.plot(mm_in.get("events")["times"]/delta_t, mm_in.get("events")['V_m.s'], label="NEST computed")
ax0.plot(UI, label="analytical")

ax1.plot(mm_h.get("events")["times"]/delta_t, mm_h.get("events")['V_m.s'], label="NEST computed")
ax1.plot(UH, label="analytical")

ax2.plot(mm_h.get("events")['times']/delta_t, mm_h.get("events")['V_m.b'], label="NEST computed")
ax2.plot(UBH, label="analytical")

ax3.plot(wr.get("events")["times"]/delta_t, wr.get("events")["weights"], label="NEST computed")
ax3.plot(W0, label="analytical")

ax0.set_title("input neuron voltage")
ax1.set_title("output neuron somatic voltage")
ax2.set_title("output neuron basal voltage")

ax0.legend()
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()