import nest
import matplotlib.pyplot as plt
from params_rate_test import *
import pandas as pd

"""
This script shows that the neuron model handles membrane voltage updates exactly as described in the analytical case.
The neuron in the hidden layer (pyr_h) recieves synaptic input to both apical and basal dendrites, as well as a direct
somatic current connection from one of the input neurons.
"""
nest.SetDefaults("multimeter", {'interval': 0.1})

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


eta = 0.03
w0 = 1
syn_hx.update({"weight": w0, "eta": eta})
nest.Connect(pyr_x, pyr_h, syn_spec=syn_hx)

w1 = -0.5
syn_hy.update({"weight": w1, "eta": eta})
nest.Connect(pyr_y, pyr_h, syn_spec=syn_hy)


U_x = 0
r_x = 0
U_y = 0
r_y = 0
U_h = 0
V_bh = 0
V_ah = 0
r_h = 0

tilde_w_0 = 0
tilde_w_1 = 0

sim_times = [250 for i in range(3)]
stim_amps = [2, 1, 0.5]
target_amps = [0.1, 0.5, 1]

SIM_TIME = sum(sim_times)


UX = [0]
UH = [0]
UY = [0]
VBH = [0]
VAH = [0]
W0 = [w0]
W1 = [w1]

for T, stim, target in zip(sim_times, stim_amps, target_amps):
    pyr_x.set({"soma": {"I_e": stim*tau_input}})
    pyr_y.set({"soma": {"I_e": phi_inverse(target) * g_s}})
    nest.Simulate(T)

    for i in range(int(T/delta_t)):

        delta_u_x = -U_x + stim
        delta_u_h = -(g_l + g_d + g_a) * U_h + V_bh * g_d + V_ah * g_a
        delta_u_y = -(g_l + g_d + g_a) * U_y + g_s * phi_inverse(target)

        vw_star = phi((g_d * V_bh)/(g_l + g_d + g_a))
        dend_error = phi(U_h) - vw_star
        delta_tilde_w = -tilde_w_0 + dend_error * phi(U_x)
        tilde_w_0 = tilde_w_0 + (delta_t * delta_tilde_w) / tau_delta
        w0 = w0 + eta * delta_t * tilde_w_0

        dend_error_2 = -V_ah
        delta_tilde_w = -tilde_w_1 + dend_error_2 * phi(U_y)
        tilde_w_1 = tilde_w_1 + (delta_t * delta_tilde_w) / tau_delta
        w1 = w1 + eta * delta_t * tilde_w_1

        U_x = U_x + (delta_t/tau_x) * delta_u_x
        r_x = phi(U_x)

        V_bh = r_x * w0
        V_ah = r_y * w1
        U_h = U_h + delta_t * delta_u_h

        U_y = U_y + delta_t * delta_u_y
        r_y = phi(U_y)

        UX.append(U_x)
        UH.append(U_h)
        UY.append(U_y)
        VBH.append(V_bh)
        VAH.append(V_ah)
        W0.append(w0)
        W1.append(w1)


fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 7, sharex=True)

ax0.plot(mm_x.get("events")["times"]/delta_t, mm_x.get("events")['V_m.s'], label="NEST computed")
ax0.plot(UX, label="analytical")

ax1.plot(mm_h.get("events")['times']/delta_t, mm_h.get("events")['V_m.b'], label="NEST computed")
ax1.plot(VBH, label="analytical")

ax2.plot(mm_h.get("events")["times"]/delta_t, mm_h.get("events")['V_m.s'], label="NEST computed")
ax2.plot(UH, label="analytical")

ax3.plot(mm_h.get("events")['times']/delta_t, mm_h.get("events")['V_m.a_lat'], label="NEST computed")
ax3.plot(VAH, label="analytical")

ax4.plot(mm_y.get("events")["times"]/delta_t, mm_y.get("events")['V_m.s'], label="NEST computed")
ax4.plot(UY, label="analytical")


nest_weights = pd.DataFrame.from_dict(wr.get("events"))

ax5.plot(nest_weights[nest_weights.senders == pyr_x.get("global_id")].weights.values)
ax5.plot(W0)

ax6.plot(nest_weights[nest_weights.senders == pyr_y.get("global_id")].weights.values)
ax6.plot(W1)


ax0.set_title("input layer voltage")
ax1.set_title("hidden layer basal voltage")
ax2.set_title("hidden layer somatic voltage")
ax3.set_title("hidden layer apical voltage")
ax4.set_title("output layer voltage")
ax5.set_title("ff weight")
ax6.set_title("fb weight")

ax0.legend()
ax1.legend()
ax2.legend()
plt.show()
