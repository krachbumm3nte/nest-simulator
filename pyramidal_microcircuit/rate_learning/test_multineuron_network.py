import nest
import matplotlib.pyplot as plt
import pandas as pd
from params_rate import *
import numpy as np
from network_rate import Network
from network_mathematica import MathematicaNetwork
from sklearn.metrics import mean_squared_error as mse


cmap = plt.cm.get_cmap('hsv', 7)
styles = ["solid", "dotted", "dashdot", "dashed"]

tau_x = 3
input_filter = 1/tau_x

n_runs = 35
SIM_TIME = 150

nest_net = Network()
math_net = MathematicaNetwork()

c_hx = nest.GetConnections(nest_net.pyr_pops[0], nest_net.pyr_pops[1])
c_yh = nest.GetConnections(nest_net.pyr_pops[1], nest_net.pyr_pops[2])
c_ih = nest.GetConnections(nest_net.pyr_pops[1], nest_net.intn_pops[0])
c_hi = nest.GetConnections(nest_net.intn_pops[0], nest_net.pyr_pops[1])
c_hy = nest.GetConnections(nest_net.pyr_pops[2], nest_net.pyr_pops[1])
nest_conns = [c_hx, c_yh, c_ih, c_hi, c_hy, ]

"""
math_net.conns["hx"]["w"] = np.asmatrix(np.ones((dims[1], dims[0])))
math_net.conns["yh"]["w"] = np.asmatrix(np.ones((dims[2], dims[1])))
math_net.conns["ih"]["w"] = np.asmatrix(np.ones((dims[2], dims[1])))
math_net.conns["hi"]["w"] = np.asmatrix(np.ones((dims[1], dims[2])))
math_net.conns["hy"]["w"] = np.asmatrix(np.ones((dims[1], dims[2])))

c_hx.set({"weight": np.squeeze(np.asarray(math_net.conns["hx"]["w"])).flatten("F"), "eta": math_net.conns["hx"]["eta"]})
c_yh.set({"weight": np.squeeze(np.asarray(math_net.conns["yh"]["w"])).flatten("F"), "eta": math_net.conns["yh"]["eta"]})
c_ih.set({"weight": np.squeeze(np.asarray(math_net.conns["ih"]["w"])).flatten("F"), "eta": math_net.conns["ih"]["eta"]})
c_hi.set({"weight": np.squeeze(np.asarray(math_net.conns["hi"]["w"])).flatten("F"), "eta": math_net.conns["hi"]["eta"]})
c_hy.set({"weight": np.squeeze(np.asarray(math_net.conns["hy"]["w"])).flatten("F"), "eta": math_net.conns["hy"]["eta"]})

math_net.conns["hx"]["w"] = np.asmatrix(np.random.random((dims[1], dims[0])) * 2 - 1)
math_net.conns["yh"]["w"] = np.asmatrix(np.random.random((dims[2], dims[1])) * 2 - 1)
math_net.conns["ih"]["w"] = np.asmatrix(np.random.random((dims[2], dims[1])) * 2 - 1)
math_net.conns["hi"]["w"] = np.asmatrix(np.random.random((dims[1], dims[2])) * 2 - 1)
math_net.conns["hy"]["w"] = np.asmatrix(np.random.random((dims[1], dims[2])) * 2 - 1)
"""


def matrix_from_connection(conn):
    df = pd.DataFrame.from_dict(conn.get(["weight", "source", "target"]))
    n_out = len(set(df["target"]))
    n_in = len(set(df["source"]))
    weights = np.reshape(df.sort_values(by=["source", "target"])["weight"].values, (n_out, n_in), "F")
    return np.asmatrix(weights)


def matrix_from_wr(data, conn):
    n_out = len(set(conn.get("target")))
    n_in = len(set(conn.get("source")))
    filtered_data = data[(data.targets.isin(set(conn.target)) & data.senders.isin(set(conn.source)))]
    sorted_data = filtered_data.sort_values(by=["senders", "targets"])["weights"].values
    return np.reshape(sorted_data, (-1, n_out, n_in), "F")


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

fig, axes = plt.subplots(2, 6, sharey="col")


WIH = np.array(math_net.conns["ih"]["record"])
WYH = np.array(math_net.conns["yh"]["record"])
WHI = np.array(math_net.conns["hi"]["record"])
WHY = np.array(math_net.conns["hy"]["record"])

ff_weight_error = np.square(WYH - WIH).mean(axis=(1, 2))
fb_weight_error = np.square(WHI + WHY).mean(axis=(1, 2))


nest_weights = pd.DataFrame.from_dict(wr.get("events"))

WIH_nest = matrix_from_wr(nest_weights, c_ih)
WYH_nest = matrix_from_wr(nest_weights, c_yh)
WHI_nest = matrix_from_wr(nest_weights, c_hi)
WHY_nest = matrix_from_wr(nest_weights, c_hy)

ff_weight_error_nest = np.square(WYH_nest - WIH_nest).mean(axis=(1, 2))
fb_weight_error_nest = np.square(WHI_nest + WHY_nest).mean(axis=(1, 2))

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
axes[0][4].plot(np.linalg.norm(np.array(math_net.V_ah_record), axis=(1,2)))
axes[0][4].set_title("apical voltage")

events = pd.DataFrame.from_dict(nest_net.mm_h.events).sort_values("times")
apical_err = events["V_m.a_lat"].values.reshape(-1,dims[1])
axes[1][4].plot(np.linalg.norm(apical_err, axis=1), label="apical error")



# plot interneuron error

U_intn = np.array(math_net.U_i_record).squeeze().swapaxes(0,1)
U_out = np.array(math_net.U_y_record).squeeze().swapaxes(0,1)


intn_error = mse(U_intn, U_out, multioutput="raw_values")

axes[0][5].plot(intn_error)
axes[0][5].set_title("interneuron error")

U_intn = pd.DataFrame.from_dict(nest_net.mm_i.events).sort_values(["times", "senders"])
U_intn = np.reshape(U_intn["V_m.s"].values, (-1,dims[2])).swapaxes(0,1)

U_out = pd.DataFrame.from_dict(nest_net.mm_y.events).sort_values(["times", "senders"])
U_out = np.reshape(U_out["V_m.s"].values, (-1,dims[2])).swapaxes(0,1)

axes[1][5].plot(mse(U_out, U_intn, multioutput="raw_values"))


plt.show()
