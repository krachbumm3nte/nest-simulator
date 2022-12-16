import nest
import matplotlib.pyplot as plt
import pandas as pd
from params_rate import *
import numpy as np
from network_rate import Network
from network_mathematica import MathematicaNetwork

cmap = plt.cm.get_cmap('hsv', 7)
styles = ["solid", "dotted", "dashdot", "dashed", "loosely dashed"]

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


def extract_nest_weights(conn):
    df = pd.DataFrame.from_dict(conn.get(["weight", "source", "target"]))
    n_out = len(set(df["target"]))
    n_in = len(set(df["source"]))
    weights = np.reshape(df.sort_values(by=["source", "target"])["weight"].values, (n_out, n_in), "F")
    return np.asmatrix(weights)


math_net.conns["hx"]["w"] = extract_nest_weights(c_hx)
math_net.conns["yh"]["w"] = extract_nest_weights(c_yh)
math_net.conns["ih"]["w"] = extract_nest_weights(c_ih)
math_net.conns["hi"]["w"] = extract_nest_weights(c_hi)
math_net.conns["hy"]["w"] = extract_nest_weights(c_hy)

nest_conns = [c_hx, c_yh, c_ih, c_hi, c_hy, ]

for i in range(n_runs):
    if i % 25 == 0:
        print(f"simulating run: {i}")
    amp = np.random.random(dims[0]) * 2 - 1

    nest_net.set_input(amp)
    nest.Simulate(SIM_TIME)

    math_net.set_input([amp])
    math_net.simulate(SIM_TIME)

fig, axes = plt.subplots(2, 5, sharey=True, sharex=True)

nest_weights = pd.DataFrame.from_dict(wr.get("events"))

for c, (name, conn) in enumerate(math_net.conns.items()):
    record = np.array(conn["record"])
    n_out, n_in = record.shape[1:]
    axes[0][c].set_title(name)

    nest_conn = nest_conns[c]
    nest_senders = sorted(set(nest_conn.get("source")))
    nest_targets = sorted(set(nest_conn.get("target")))

    for i in range(n_in):
        for j in range(n_out):
            col = cmap(i)
            axes[0][c].plot(record[:, j, i], color=col, linestyle=styles[j], label=f"{i} to {j}")
            # axes[0][c].legend()

            s = nest_senders[i]
            t = nest_targets[j]
            df_weight = nest_weights[(nest_weights.senders == s) & (
                nest_weights.targets == t)].sort_values(by=['times'])
            axes[1][c].plot(df_weight['times'].array/delta_t, df_weight['weights'],
                            color=col, linestyle=styles[j], label=f"{s} to {t}")
            # axes[1][c].legend()


plt.show()
