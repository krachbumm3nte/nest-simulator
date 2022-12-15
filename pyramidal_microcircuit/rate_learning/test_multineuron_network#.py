import nest
import matplotlib.pyplot as plt
import pandas as pd
from params_rate import *
import numpy as np
from network_rate import Network
from network_mathematica import MathematicaNetwork

cmap = plt.cm.get_cmap('hsv', 5)


tau_x = 3
input_filter = 1/tau_x

n_runs = 3
SIM_TIME = 100

nest_net = Network()
math_net = MathematicaNetwork()

c_hx = nest.GetConnections(nest_net.pyr_pops[0], nest_net.pyr_pops[1])
c_yh = nest.GetConnections(nest_net.pyr_pops[1], nest_net.pyr_pops[2])
c_ih = nest.GetConnections(nest_net.pyr_pops[1], nest_net.intn_pops[0])
c_hi = nest.GetConnections(nest_net.intn_pops[0], nest_net.pyr_pops[1])
c_hy = nest.GetConnections(nest_net.pyr_pops[2], nest_net.pyr_pops[1])

math_net.conns["hx"]["w"] = np.asmatrix(np.ones((dims[1], dims[0])))
math_net.conns["yh"]["w"] = np.asmatrix(np.ones((dims[2], dims[1])))
math_net.conns["ih"]["w"] = np.asmatrix(np.ones((dims[2], dims[1])))
math_net.conns["hi"]["w"] = np.asmatrix(np.ones((dims[1], dims[2])))
math_net.conns["hy"]["w"] = np.asmatrix(np.ones((dims[1], dims[2])))


c_hx.set({"weight": np.squeeze(np.asarray(math_net.conns["hx"]["w"])).flatten(), "eta": math_net.conns["hx"]["eta"]})
c_yh.set({"weight": np.squeeze(np.asarray(math_net.conns["yh"]["w"])).flatten(), "eta": math_net.conns["yh"]["eta"]})
c_ih.set({"weight": np.squeeze(np.asarray(math_net.conns["ih"]["w"])).flatten(), "eta": math_net.conns["ih"]["eta"]})
c_hi.set({"weight": np.squeeze(np.asarray(math_net.conns["hi"]["w"])).flatten(), "eta": math_net.conns["hi"]["eta"]})
c_hy.set({"weight": np.squeeze(np.asarray(math_net.conns["hy"]["w"])).flatten(), "eta": math_net.conns["hy"]["eta"]})

nest_conns = [c_hx, c_yh, c_ih, c_hi, c_hy, ]

for i in range(n_runs):
    amp = np.random.random(dims[0]) * 2 - 1

    nest_net.set_input(amp)
    nest.Simulate(SIM_TIME)

    math_net.set_input([amp])
    math_net.simulate(SIM_TIME)

fig, axes = plt.subplots(2, 5, sharey=True, sharex=True)

weights_nest = pd.DataFrame.from_dict(wr.events)


axes[0][0].plot(nest_net.mm_x.get("events")["times"]/delta_t,
                nest_net.mm_x.get("events")['V_m.s'], label="NEST computed")
axes[0][0].plot(math_net.UX, label="analytical")
axes[0][0].set_title("UX")

axes[0][1].plot(nest_net.mm_h.get("events")["times"]/delta_t,
                nest_net.mm_h.get("events")['V_m.s'], label="NEST computed")
axes[0][1].plot(math_net.UH, label="analytical")
axes[0][1].set_title("UH")

axes[0][2].plot(nest_net.mm_h.get("events")["times"]/delta_t,
                nest_net.mm_h.get("events")['V_m.b'], label="NEST computed")
axes[0][2].plot(math_net.VBH, label="analytical")
axes[0][2].set_title("VBH")

axes[0][3].plot(nest_net.mm_h.get("events")["times"]/delta_t,
                nest_net.mm_h.get("events")['V_m.a_lat'], label="NEST computed")
axes[0][3].plot(math_net.VAH, label="analytical")
axes[0][3].set_title("VAH")

axes[1][0].plot(nest_net.mm_i.get("events")["times"]/delta_t,
                nest_net.mm_i.get("events")['V_m.s'], label="NEST computed")
axes[1][0].plot(math_net.UI, label="analytical")
axes[1][0].set_title("UI")

axes[1][1].plot(nest_net.mm_i.get("events")["times"]/delta_t,
                nest_net.mm_i.get("events")['V_m.b'], label="NEST computed")
axes[1][1].plot(math_net.VBI, label="analytical")
axes[1][1].set_title("VBI")

axes[1][2].plot(nest_net.mm_y.get("events")["times"]/delta_t,
                nest_net.mm_y.get("events")['V_m.s'], label="NEST computed")
axes[1][2].plot(math_net.UY, label="analytical")
axes[1][2].set_title("UY")

axes[1][3].plot(nest_net.mm_y.get("events")["times"]/delta_t,
                nest_net.mm_y.get("events")['V_m.b'], label="NEST computed")
axes[1][3].plot(math_net.VBY, label="analytical")
axes[1][3].set_title("VBY")

for i, (k, v) in enumerate(math_net.conns.items()):
    axes[0][4].plot(v["record"], color=cmap(i), label=k)
axes[0][4].set_title("analytical weights")
axes[0][4].legend()

why = -math_net.conns["hy"]["w"]
wih = (g_d + g_l)/(g_d + g_a + g_l) * math_net.conns["yh"]["w"]

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
