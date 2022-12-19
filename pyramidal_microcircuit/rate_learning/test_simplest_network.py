import nest
import matplotlib.pyplot as plt
import pandas as pd
from params_rate import *
from utils import *
import numpy as np
from network_rate import Network
from network_mathematica import MathematicaNetwork
from sklearn.metrics import mean_squared_error as mse

cmap = plt.cm.get_cmap('hsv', 7)
styles = ["solid", "dotted", "dashdot", "dashed"]

n_runs = 5
SIM_TIME = 130
SIM_TIME_TOTAL = n_runs * SIM_TIME

dims = [1, 1, 1]

nest_net = Network(dims)
math_net = MathematicaNetwork(dims)

c_hx = nest.GetConnections(nest_net.pyr_pops[0], nest_net.pyr_pops[1])
c_yh = nest.GetConnections(nest_net.pyr_pops[1], nest_net.pyr_pops[2])
c_ih = nest.GetConnections(nest_net.pyr_pops[1], nest_net.intn_pops[0])
c_hi = nest.GetConnections(nest_net.intn_pops[0], nest_net.pyr_pops[1])
c_hy = nest.GetConnections(nest_net.pyr_pops[2], nest_net.pyr_pops[1])
nest_conns = [c_hx, c_yh, c_ih, c_hi, c_hy, ]

math_net.conns["hx"]["w"] = matrix_from_connection(c_hx)
math_net.conns["yh"]["w"] = matrix_from_connection(c_yh)
math_net.conns["ih"]["w"] = matrix_from_connection(c_ih)
math_net.conns["hi"]["w"] = matrix_from_connection(c_hi)
math_net.conns["hy"]["w"] = matrix_from_connection(c_hy)


for i in range(n_runs):
    if i % 5 == 0:
        print(f"simulating run: {i}")
    amp = np.random.random(dims[0]) * 2 - 1

    nest_net.set_input(amp)
    nest.Simulate(SIM_TIME)

    math_net.set_input([amp])
    math_net.simulate(SIM_TIME)

fig, axes = plt.subplots(2, 5, sharex=True)

axes[0][0].plot(nest_net.mm_x.get("events")["times"]/delta_t, nest_net.mm_x.get("events")['V_m.s'], label="NEST computed")
axes[0][0].plot(np.asarray(math_net.U_x_record).squeeze(), label="analytical")
axes[0][0].legend()
axes[0][0].set_title("UX")

axes[0][1].plot(nest_net.mm_h.get("events")["times"]/delta_t, nest_net.mm_h.get("events")['V_m.s'], label="NEST computed")
axes[0][1].plot(np.asarray(math_net.U_h_record).squeeze(), label="analytical")
axes[0][1].set_title("UH")

axes[0][2].plot(nest_net.mm_h.get("events")["times"]/delta_t, nest_net.mm_h.get("events")['V_m.b'], label="NEST computed")
axes[0][2].plot(np.asarray(math_net.V_bh_record).squeeze(), label="analytical")
axes[0][2].set_title("VBH")

axes[0][3].plot(nest_net.mm_h.get("events")["times"]/delta_t, nest_net.mm_h.get("events")['V_m.a_lat'], label="NEST computed")
axes[0][3].plot(np.asarray(math_net.V_ah_record).squeeze(), label="analytical")
axes[0][3].set_title("VAH")

axes[1][0].plot(nest_net.mm_i.get("events")["times"]/delta_t, nest_net.mm_i.get("events")['V_m.s'], label="NEST computed")
axes[1][0].plot(np.asarray(math_net.U_i_record).squeeze(), label="analytical")
axes[1][0].set_title("UI")

axes[1][1].plot(nest_net.mm_i.get("events")["times"]/delta_t, nest_net.mm_i.get("events")['V_m.b'], label="NEST computed")
axes[1][1].plot(np.asarray(math_net.V_bi_record).squeeze(), label="analytical")
axes[1][1].set_title("VBI")

axes[1][2].plot(nest_net.mm_y.get("events")["times"]/delta_t, nest_net.mm_y.get("events")['V_m.s'], label="NEST computed")
axes[1][2].plot(np.asarray(math_net.U_y_record).squeeze(), label="analytical")
axes[1][2].set_title("UY")

axes[1][3].plot(nest_net.mm_y.get("events")["times"]/delta_t, nest_net.mm_y.get("events")['V_m.b'], label="NEST computed")
axes[1][3].plot(np.asarray(math_net.V_by_record).squeeze(), label="analytical")
axes[1][3].set_title("VBY")

for i, (k, v) in enumerate(math_net.conns.items()):
    axes[0][4].plot(v["record"].squeeze(), color=cmap(i), label=k)
axes[0][4].set_title("analytical weights")
axes[0][4].legend()

why = -math_net.conns["hy"]["w"].squeeze()
wih = (g_d + g_l)/(g_d + g_a + g_l) * math_net.conns["yh"]["w"].squeeze()

axes[0][4].hlines([why, wih], 0, SIM_TIME_TOTAL/delta_t)


wgts = pd.DataFrame.from_dict(wr.get("events"))
for i, conn in enumerate(nest_conns):
    s = conn.source
    t = conn.target
    df_weight = wgts[(wgts.senders == s) & (wgts.targets == t)].sort_values(by=['times'])
    axes[1][4].plot(df_weight['times'].array/delta_t, df_weight['weights'], color=cmap(i))

axes[1][4].hlines([why, wih], 0, SIM_TIME_TOTAL/delta_t)
axes[1][4].set_title("NEST weights")

plt.show()
