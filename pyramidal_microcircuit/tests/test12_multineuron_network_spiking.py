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


dims = [4, 3, 2]
imgdir, datadir = setup_simulation()
sim_params["record_interval"] = 0.1
sim_params["noise"] = False
sim_params["dims"] = dims
setup_nest(delta_t, sim_params["threads"], sim_params["record_interval"], datadir)
wr = setup_models(True, True)

cmap = plt.cm.get_cmap('hsv', 7)
styles = ["solid", "dotted", "dashdot", "dashed"]

n_runs = 25
SIM_TIME = 250
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
for i in range(n_runs):
    amp = np.random.random(size=dims[0])
    nest_net.set_input(amp)
    nest_net.simulate(SIM_TIME)

    math_net.set_input(amp)
    math_net.train(SIM_TIME)

fig, axes = plt.subplots(2, 6, sharey="col")


WIH = np.array(math_net.conns["ih"]["record"])
WYH = np.array(math_net.conns["yh"]["record"])
WHI = np.array(math_net.conns["hi"]["record"])
WHY = np.array(math_net.conns["hy"]["record"])

ff_weight_error = np.square(WYH - WIH).mean(axis=(1, 2))
fb_weight_error = np.square(WHI + WHY).mean(axis=(1, 2))


nest_weights = pd.DataFrame.from_dict(wr.get("events"))
nest_weights.times = (nest_weights.times // delta_t).astype(int)

WIH_nest = matrix_from_spikes(nest_weights, c_ih, SIM_TIME_TOTAL, delta_t)
WYH_nest = matrix_from_spikes(nest_weights, c_yh, SIM_TIME_TOTAL, delta_t)
WHI_nest = matrix_from_spikes(nest_weights, c_hi, SIM_TIME_TOTAL, delta_t)
WHY_nest = matrix_from_spikes(nest_weights, c_hy, SIM_TIME_TOTAL, delta_t)

ff_weight_error_nest = np.square(WYH_nest - WIH_nest).mean(axis=1)
fb_weight_error_nest = np.square(WHI_nest + WHY_nest).mean(axis=1)

axes[0][0].set_title("WYH - WIH (ff) error")
axes[0][0].set_ylim(0,)
axes[0][0].plot(ff_weight_error)
axes[1][0].plot(ff_weight_error_nest)

axes[0][1].set_title("WHI + WHY (fb) error")
axes[0][1].set_ylim(0,)
axes[0][1].plot(fb_weight_error)
axes[1][1].plot(fb_weight_error_nest)

for c, (name, conn) in enumerate(math_net.conns.items()):
    if c not in [2, 3]:
        continue
    record = np.array(conn["record"])
    n_out, n_in = record.shape[1:]
    axes[0][c].set_title(name)

    nest_conn = nest_conns[c]
    nest_senders = sorted(set(nest_conn.get("source")))
    nest_targets = sorted(set(nest_conn.get("target")))

    for i in range(n_in):
        for j in range(n_out):
            col = cmap(i)
            axes[0][c].plot(record[:, j, i], color=col, linestyle=styles[j % 4], label=f"{i} to {j}")
            # axes[0][c].legend()

            s = nest_senders[i]
            t = nest_targets[j]
            df_weight = nest_weights[(nest_weights.senders == s) & (
                nest_weights.targets == t)].sort_values(by=['times'])
            axes[1][c].plot(df_weight['times'].array/delta_t, df_weight['weights'],
                            color=col, linestyle=styles[j % 4], label=f"{s} to {t}")
            # axes[1][c].legend()

# plot apical error
axes[0][4].plot(np.linalg.norm(math_net.V_ah_record, axis=1))
axes[0][4].set_title("apical voltage")

events = pd.DataFrame.from_dict(nest_net.mm_h.events).sort_values("times")
apical_err = events["V_m.a_lat"].values.reshape(-1, dims[1])
axes[1][4].plot(np.linalg.norm(apical_err, axis=1), label="apical error")


# plot interneuron error

U_intn = np.asarray(math_net.U_i_record).swapaxes(0, 1)
U_out = np.asarray(math_net.U_y_record).swapaxes(0, 1)


intn_error = mse(U_intn, U_out, multioutput="raw_values")

axes[0][5].plot(intn_error)
axes[0][5].set_title("interneuron error")

U_intn = pd.DataFrame.from_dict(nest_net.mm_i.events).sort_values(["times", "senders"])
U_intn = np.reshape(U_intn["V_m.s"].values, (-1, dims[2])).swapaxes(0, 1)

U_out = pd.DataFrame.from_dict(nest_net.mm_y.events).sort_values(["times", "senders"])
U_out = np.reshape(U_out["V_m.s"].values, (-1, dims[2])).swapaxes(0, 1)

axes[1][5].plot(mse(U_out, U_intn, multioutput="raw_values"))


plt.show()
