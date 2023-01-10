import nest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import sys
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")

from params import *  # nopep8
from utils import *  # nopep8
from networks.network_nest import Network  # nopep8
from networks.network_numpy import NumpyNetwork  # nopep8


imgdir, datadir = setup_simulation()
sim_params["record_interval"] = 0.1
sim_params["noise"] = False
sim_params["dims"] = [1, 1, 1]
setup_nest(delta_t, sim_params["threads"], sim_params["record_interval"], datadir)
wr = setup_models(True, True)

cmap = plt.cm.get_cmap('hsv', 7)
styles = ["solid", "dotted", "dashdot", "dashed"]

amps = [0.5, 1, 0]
n_runs = len(amps)
SIM_TIME = 1000
SIM_TIME_TOTAL = n_runs * SIM_TIME


math_net = NumpyNetwork(sim_params, neuron_params, syn_params)

syn_params["hi"]["eta"] *= 0.001
syn_params["ih"]["eta"] *= 0.003

weight_scale = 45
weight_scale = 1
syn_params["wmin_init"] = -1/weight_scale
syn_params["wmax_init"] = 1/weight_scale

g_lk_dnd = 0.0023
g_lk_dnd = 0.095
neuron_params["pyr"]["basal"]["g_L"] = g_lk_dnd
neuron_params["pyr"]["apical_lat"]["g_L"] = g_lk_dnd
neuron_params["intn"]["basal"]["g_L"] = g_lk_dnd


nest_net = Network(sim_params, neuron_params, syn_params)

c_hx = nest.GetConnections(nest_net.pyr_pops[0], nest_net.pyr_pops[1])
c_yh = nest.GetConnections(nest_net.pyr_pops[1], nest_net.pyr_pops[2])
c_ih = nest.GetConnections(nest_net.pyr_pops[1], nest_net.intn_pops[0])
c_hi = nest.GetConnections(nest_net.intn_pops[0], nest_net.pyr_pops[1])
c_hy = nest.GetConnections(nest_net.pyr_pops[2], nest_net.pyr_pops[1])
nest_conns = [c_hx, c_yh, c_ih, c_hi, c_hy, ]


math_net.conns["hx"]["w"] = matrix_from_connection(c_hx) * weight_scale
math_net.conns["yh"]["w"] = matrix_from_connection(c_yh) * weight_scale
math_net.conns["ih"]["w"] = matrix_from_connection(c_ih) * weight_scale
math_net.conns["hi"]["w"] = matrix_from_connection(c_hi) * weight_scale
math_net.conns["hy"]["w"] = matrix_from_connection(c_hy) * weight_scale


nest_net.mm_x = nest.Create('multimeter', 1, {'record_to': 'memory', 'record_from': ["V_m.s"]})
nest.Connect(nest_net.mm_x, nest_net.pyr_pops[0])
nest_net.mm_h = nest.Create('multimeter', 1, {'record_to': 'memory', 'record_from': ["V_m.s", "V_m.b", "V_m.a_lat"]})
nest.Connect(nest_net.mm_h, nest_net.pyr_pops[1])
nest_net.mm_i = nest.Create('multimeter', 1, {'record_to': 'memory', 'record_from': ["V_m.s", "V_m.b"]})
nest.Connect(nest_net.mm_i, nest_net.intn_pops[0])
nest_net.mm_y = nest.Create('multimeter', 1, {'record_to': 'memory', 'record_from': ["V_m.s", "V_m.b"]})
nest.Connect(nest_net.mm_y, nest_net.pyr_pops[-1])

print("Setup complete.")
# for i in range(n_runs):
# if i % 5 == 0:
# print(f"simulating run: {i}")
# amp = np.random.random(sim_params["dims"][0])
for amp in amps:
    amp = [amp]
    nest_net.set_input(amp)
    nest_net.simulate(SIM_TIME)

    math_net.set_input(amp)
    math_net.train(SIM_TIME)

print("Simulation complete")

fig, axes = plt.subplots(2, 5, sharex=True)

filtersize = 2500

axes[0][0].plot(nest_net.mm_x.get("events")["times"]/delta_t,
                nest_net.mm_x.get("events")['V_m.s'], label="NEST computed")
axes[0][0].plot(np.asarray(math_net.U_x_record).squeeze(), label="analytical")
axes[0][0].legend()
axes[0][0].set_title("UX")

axes[0][1].plot(nest_net.mm_h.get("events")["times"]/delta_t,
                nest_net.mm_h.get("events")['V_m.s'], label="NEST computed", alpha=0.3)
axes[0][1].plot(nest_net.mm_h.get("events")["times"]/delta_t,
                rolling_avg(nest_net.mm_h.get("events")['V_m.s'], size=filtersize), label="rolling average")
axes[0][1].plot(np.asarray(math_net.U_h_record).squeeze(), label="analytical")
axes[0][1].set_title("UH")

axes[0][2].plot(nest_net.mm_h.get("events")["times"]/delta_t,
                nest_net.mm_h.get("events")['V_m.b'], label="NEST computed", alpha=0.3)
axes[0][2].plot(nest_net.mm_h.get("events")["times"]/delta_t,
                rolling_avg(nest_net.mm_h.get("events")['V_m.b'], size=filtersize), label="rolling average")
axes[0][2].plot(np.asarray(math_net.V_bh_record).squeeze(), label="analytical")
axes[0][2].set_title("VBH")

axes[0][3].plot(nest_net.mm_h.get("events")["times"]/delta_t,
                nest_net.mm_h.get("events")['V_m.a_lat'], label="NEST computed", alpha=0.3)
axes[0][3].plot(nest_net.mm_h.get("events")["times"]/delta_t,
                rolling_avg(nest_net.mm_h.get("events")['V_m.a_lat'], size=filtersize), label="rolling average")
axes[0][3].plot(np.asarray(math_net.V_ah_record).squeeze(), label="analytical")
axes[0][3].set_title("VAH")

axes[1][0].plot(nest_net.mm_i.get("events")["times"]/delta_t,
                nest_net.mm_i.get("events")['V_m.s'], label="NEST computed", alpha=0.3)
axes[1][0].plot(nest_net.mm_i.get("events")["times"]/delta_t,
                rolling_avg(nest_net.mm_i.get("events")['V_m.s'], size=filtersize), label="rolling average")
axes[1][0].plot(np.asarray(math_net.U_i_record).squeeze(), label="analytical")
axes[1][0].set_title("UI")

axes[1][1].plot(nest_net.mm_i.get("events")["times"]/delta_t,
                nest_net.mm_i.get("events")['V_m.b'], label="NEST computed", alpha=0.3)
axes[1][1].plot(nest_net.mm_i.get("events")["times"]/delta_t,
                rolling_avg(nest_net.mm_i.get("events")['V_m.b'], size=filtersize), label="rolling average")
axes[1][1].plot(np.asarray(math_net.V_bi_record).squeeze(), label="analytical")
axes[1][1].set_title("VBI")

axes[1][2].plot(nest_net.mm_y.get("events")["times"]/delta_t,
                nest_net.mm_y.get("events")['V_m.s'], label="NEST computed", alpha=0.3)
axes[1][2].plot(nest_net.mm_y.get("events")["times"]/delta_t,
                rolling_avg(nest_net.mm_y.get("events")['V_m.s'], size=filtersize), label="rolling average")
axes[1][2].plot(np.asarray(math_net.U_y_record).squeeze(), label="analytical")
axes[1][2].set_title("UY")

axes[1][3].plot(nest_net.mm_y.get("events")["times"]/delta_t,
                nest_net.mm_y.get("events")['V_m.b'], label="NEST computed", alpha=0.3)
axes[1][3].plot(nest_net.mm_y.get("events")["times"]/delta_t,
                rolling_avg(nest_net.mm_y.get("events")['V_m.b'], size=filtersize), label="rolling average")
axes[1][3].plot(np.asarray(math_net.V_by_record).squeeze(), label="analytical")
axes[1][3].set_title("VBY")

wgts = pd.DataFrame.from_dict(wr.get("events"))
for i, (k, v) in enumerate(math_net.conns.items()):
    math_weight = v["record"].squeeze() / weight_scale
    axes[0][4].plot(math_weight, color=cmap(i), label=k)

    conn = nest_conns[i]
    s = conn.source
    t = conn.target
    df_weight = wgts[(wgts.senders == s) & (wgts.targets == t)].sort_values(by=['times'])
    nest_weight = df_weight['weights']

    times = (df_weight['times'].array / delta_t).astype(int) - 1

    axes[0][4].plot(times, nest_weight, color=cmap(i), linestyle="--")

    axes[1][4].plot(times, nest_weight - math_weight[times], color=cmap(i))

axes[0][4].hlines(y=[- math_net.conns["hy"]["w"], math_net.conns["yh"]["w"] * 1.1 / 1.9], xmin=0, xmax=SIM_TIME_TOTAL/delta_t, linestyles="dashed", colors=["grey", "grey"], alpha=0.4)

axes[0][4].set_title("weights: nest(--), other(-)")
axes[0][4].legend()
axes[1][4].set_title("nest weights - math weights")

plt.show()
