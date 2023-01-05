import nest
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from sklearn.metrics import mean_squared_error as mse
sys.path.append("..")
# sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")
from pyramidal_microcircuit.params.params_rate import *  # nopep8
from pyramidal_microcircuit.utils import *  # nopep8
from pyramidal_microcircuit.networks.network_rate import Network  # nopep8
from pyramidal_microcircuit.networks.network_mathematica import MathematicaNetwork  # nopep8

cmap = plt.cm.get_cmap('hsv', 7)
styles = ["solid", "dotted", "dashdot", "dashed"]

n_runs = 25
SIM_TIME = 200
SIM_TIME_TOTAL = n_runs * SIM_TIME

def phi(x):
    return 1 / (1.0 + np.exp(-x))


tau_x = 3
input_filter = 1/tau_x

n_runs = 500
sim_duration = 100
sim_times = [sim_duration for i in range(n_runs)]
stim_amps = 2*np.random.random(n_runs)-1
SIM_TIME = sum(sim_times)

# rates, somatic and dendritic voltages for all neurons
U_x, r_x, U_h, V_bh, V_ah, r_h, U_i, V_bi, r_i, V_by, U_y, r_y = [0 for i in range(12)]

# lists to store neuron states
UX, UH, VBH, VAH, UI, VBI, UY, VBY = [[] for i in range(8)]

net = Network()

conn_names = ["hx", "yh", "ih", "hi", "hy"]

conns = {n: {"tilde_w": 0, "delta_tilde_w": 0, "record": [], "eta": 0, "w": 1} for n in conn_names}

conns["hi"]["eta"] = 0.003
conns["ih"]["eta"] = 0.003

c_hx = nest.GetConnections(net.pyr_pops[0], net.pyr_pops[1])
c_yh = nest.GetConnections(net.pyr_pops[1], net.pyr_pops[2])
c_ih = nest.GetConnections(net.pyr_pops[1], net.intn_pops[0])
c_hi = nest.GetConnections(net.intn_pops[0], net.pyr_pops[1])
c_hy = nest.GetConnections(net.pyr_pops[2], net.pyr_pops[1])


c_hx.set({"weight": conns["hx"]["w"], "eta": conns["hx"]["eta"]})
c_yh.set({"weight": conns["yh"]["w"], "eta": conns["yh"]["eta"]})
c_ih.set({"weight": conns["ih"]["w"], "eta": conns["ih"]["eta"]})
c_hi.set({"weight": conns["hi"]["w"], "eta": conns["hi"]["eta"]})
c_hy.set({"weight": conns["hy"]["w"], "eta": conns["hy"]["eta"]})

nest_conns = [c_hx, c_yh, c_ih, c_hi, c_hy, ]

for T, amp in zip(sim_times, stim_amps):
    net.set_input([amp])
    nest.Simulate(T)

    for i in range(int(T/delta_t)):

        delta_u_x = -U_x + amp
        delta_u_h = -(g_l + g_d + g_a) * U_h + g_d * V_bh + g_a * V_ah
        delta_u_y = -(g_l + g_d + g_a) * U_y + g_d * V_by  # + target activation if we want that
        delta_u_i = -(g_l + g_d + g_a) * U_i + g_d * V_bi + lam * U_y

        conns["hx"]["delta_tilde_w"] = -conns["hx"]["tilde_w"] + (r_h - phi((g_d * V_bh)/(g_l + g_d + g_a))) * r_x
        conns["yh"]["delta_tilde_w"] = -conns["yh"]["tilde_w"] + (r_y - phi((g_d * V_by)/(g_l + g_d))) * r_h
        conns["hy"]["delta_tilde_w"] = -conns["hy"]["tilde_w"] + (r_h - phi(conns["hy"]["w"] * r_y)) * r_y

        conns["ih"]["delta_tilde_w"] = -conns["ih"]["tilde_w"] + (r_i - phi((g_d * V_bi)/(g_l + g_d))) * r_h
        conns["hi"]["delta_tilde_w"] = -conns["hi"]["tilde_w"] + (-V_ah * r_i)

        U_x = U_x + (delta_t/tau_x) * delta_u_x
        r_x = phi(U_x)  # TODO: strictly speaking this should just be U_x but use_phi does not work as intended yet.

        V_bh = conns["hx"]["w"] * r_x
        V_ah = conns["hy"]["w"] * r_y + conns["hi"]["w"] * r_i
        U_h = U_h + delta_t * delta_u_h
        r_h = phi(U_h)

        V_by = conns["yh"]["w"] * r_h
        U_y = U_y + delta_t * delta_u_y
        r_y = phi(U_y)

        V_bi = conns["ih"]["w"] * r_h
        U_i = U_i + delta_t * delta_u_i
        r_i = phi(U_i)

        UX.append(U_x)

        VBH.append(V_bh)
        VAH.append(V_ah)
        UH.append(U_h)

        VBY.append(V_by)
        UY.append(U_y)

        VBI.append(V_bi)
        UI.append(U_i)

        for name, d in conns.items():
            d["tilde_w"] += (delta_t/tau_delta) * d["delta_tilde_w"]
            d["w"] += d["eta"] * delta_t * d["tilde_w"]
            d["record"].append(d["w"])


fig, axes = plt.subplots(2, 5, sharey=True, sharex=True)

axes[0][0].plot(net.mm_x.get("events")["times"]/delta_t, net.mm_x.get("events")['V_m.s'], label="NEST computed")
axes[0][0].plot(UX, label="analytical")
axes[0][0].set_title("UX")

axes[0][1].plot(net.mm_h.get("events")["times"]/delta_t, net.mm_h.get("events")['V_m.s'], label="NEST computed")
axes[0][1].plot(UH, label="analytical")
axes[0][1].set_title("UH")

axes[0][2].plot(net.mm_h.get("events")["times"]/delta_t, net.mm_h.get("events")['V_m.b'], label="NEST computed")
axes[0][2].plot(VBH, label="analytical")
axes[0][2].set_title("VBH")

axes[0][3].plot(net.mm_h.get("events")["times"]/delta_t, net.mm_h.get("events")['V_m.a_lat'], label="NEST computed")
axes[0][3].plot(VAH, label="analytical")
axes[0][3].set_title("VAH")

axes[1][0].plot(net.mm_i.get("events")["times"]/delta_t, net.mm_i.get("events")['V_m.s'], label="NEST computed")
axes[1][0].plot(UI, label="analytical")
axes[1][0].set_title("UI")

axes[1][1].plot(net.mm_i.get("events")["times"]/delta_t, net.mm_i.get("events")['V_m.b'], label="NEST computed")
axes[1][1].plot(VBI, label="analytical")
axes[1][1].set_title("VBI")

axes[1][2].plot(net.mm_y.get("events")["times"]/delta_t, net.mm_y.get("events")['V_m.s'], label="NEST computed")
axes[1][2].plot(UY, label="analytical")
axes[1][2].set_title("UY")

axes[1][3].plot(net.mm_y.get("events")["times"]/delta_t, net.mm_y.get("events")['V_m.b'], label="NEST computed")
axes[1][3].plot(VBY, label="analytical")
axes[1][3].set_title("VBY")

for i, (k, v) in enumerate(conns.items()):
    axes[0][4].plot(v["record"], color=cmap(i), label=k)
axes[0][4].set_title("analytical weights")
axes[0][4].legend()

why = -conns["hy"]["w"]
wih = (g_d + g_l)/(g_d + g_a + g_l) * conns["yh"]["w"]

ax1_2 = axes[0][4].secondary_yaxis("right")
ax1_2.set_yticks([why, wih])
axes[0][4].hlines([why, wih], 0, sum(sim_times)*10)


wgts = pd.DataFrame.from_dict(wr.get("events"))
for i, conn in enumerate(nest_conns):
    s = conn.source
    t = conn.target
    df_weight = wgts[(wgts.senders == s) & (wgts.targets == t)].sort_values(by=['times'])
    axes[1][4].plot(df_weight['times'].array/delta_t, df_weight['weights'], color=cmap(i))
ax2_2 = axes[1][4].secondary_yaxis("right")
ax2_2.set_yticks([why, wih])
axes[1][4].hlines([why, wih], 0, sum(sim_times)*10)
axes[1][4].set_title("NEST weights")

plt.show()
