import nest
import argparse
import os
import sys
import time
import json
from datetime import timedelta

import numpy as np
import src.utils as utils
import src.plot_utils as plot_utils
from src.params import Params
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd

plot_utils.setup_plt()

t_sim = 500
n = 4

linestyles = ["-", "--", ":", "-."]

col_exc = "red"
col_inh = "blue"

p = Params()
p.spiking = True
p.weight_scale = 200
p.C_m_api = 5
p.C_m_bas = 5
p.threads = 3


def phi(x, thresh=15):

    res = x.copy()
    ind = np.abs(x) < thresh
    res[x < -thresh] = 0
    res[ind] = p.gamma * np.log(1 + np.exp(p.beta * (x[ind] - p.theta)))

    return res


fig, axes = plt.subplots(2, 2, sharex=True, gridspec_kw={"wspace": 0}, figsize=[8, 5])


for i in range(2):
    utils.setup_nest(p)
    p.setup_nest_configs()

    # initialize a weight_recorder, and update all synapse models to interface with it
    wr = nest.Create("weight_recorder", params={'record_to': "memory"})

    nest.CopyModel(p.syn_model, 'record_syn', {"weight_recorder": wr})
    p.syn_model = 'record_syn'

    p.syn_plastic["synapse_model"] = "record_syn"

    input_neuron = nest.Create(p.neuron_model, n, p.pyr_params)
    interneuron = nest.Create(p.neuron_model, n, p.intn_params)
    target_neuron = nest.Create(p.neuron_model, 1, p.pyr_params)

    if i == 0:
        compname = 'apical_lat'
        p.syn_plastic["eta"] = 0.0008 / (p.weight_scale**2 * p.delta_t)
        i_e = 0.75
        err_comp = "apical_lat"
    else:
        compname = 'basal'
        p.syn_plastic["eta"] = 0.0006 / (p.weight_scale**3 * p.delta_t)
        target_neuron.set({"apical_lat": {"g": 0}})
        i_e = 0.43
        err_comp = "soma"

    comp = p.compartments[compname]
    p.syn_plastic["receptor_type"] = comp
    p.syn_static["receptor_type"] = comp

    wmax = 4
    syn_exc = deepcopy(p.syn_plastic)
    syn_exc["Wmin"] = 0
    syn_exc["Wmax"] = wmax/p.weight_scale
    syn_exc["weight"] = nest.random.uniform(0, wmax/p.weight_scale)
    syn_exc["delay"] = 2 * p.delta_t

    syn_inh = deepcopy(p.syn_plastic)
    syn_inh["Wmax"] = 0
    syn_inh["Wmin"] = -wmax/p.weight_scale
    syn_inh["weight"] = nest.random.uniform(-wmax/p.weight_scale, 0)

    syn_static = p.syn_static
    syn_static["weight"] = nest.random.uniform(min=0, max=0.3*wmax/p.weight_scale)

    nest.Connect(input_neuron, target_neuron, syn_spec=syn_exc)
    nest.Connect(input_neuron, interneuron, syn_spec=syn_static)
    nest.Connect(interneuron, target_neuron, syn_spec=syn_inh)

    mm1 = nest.Create("multimeter", 1, {'record_from': ["V_m.a_lat", "V_m.s", "V_m.b"], 'record_to': 'memory'})
    nest.Connect(mm1, target_neuron)

    for j in range(n):
        input_neuron[j].set({"soma": {"I_e": np.random.uniform(low=-0.5, high=0.5)}})

    target_neuron.set({err_comp: {"I_e": i_e}})
    nest.Simulate(t_sim)

    target_neuron.set({err_comp: {"I_e": -i_e}})
    nest.Simulate(t_sim)

    weight_record = pd.DataFrame.from_dict(wr.get("events"))
    grouped_records = weight_record.groupby(["senders", "targets"])

    in_id = input_neuron.global_id
    intn_id = interneuron.global_id
    out_id = target_neuron.global_id

    w_exc = utils.read_wr(grouped_records, input_neuron, target_neuron, 2*t_sim) * p.weight_scale
    w_inh = utils.read_wr(grouped_records, interneuron, target_neuron, 2*t_sim) * p.weight_scale

    u_som = np.array(mm1.events["V_m.s"])
    v_api = np.array(mm1.events["V_m.a_lat"])
    v_bas = np.array(mm1.events["V_m.b"])

    # plot dendritic error
    if i == 0:
        dend_error = v_api
        axes[0][i].set_title(r"Apical dendritic error ($v^{api}_c$)")

    else:
        dend_error = phi(u_som) - phi(v_bas * p.g_d / (p.g_d + p.g_l))
        axes[0][i].set_title(r"Basal dendritic error ($\phi(u^{som}_c) - \phi(\hat{v}^{bas}_c)$)")

    times = mm1.events["times"]
    axes[0][i].plot(times, dend_error, color="black")

    # plot incoming weights
    for nrn in range(n):
        axes[1][i].plot(np.arange(w_exc.shape[0])*p.delta_t, w_exc[:, 0, nrn],
                        color=col_exc, linestyle=linestyles[nrn % n])

        axes[1][i].plot(np.arange(w_inh.shape[0])*p.delta_t, w_inh[:, 0, nrn],
                        color=col_inh, linestyle=linestyles[nrn % n])

    axes[1][i].plot(0, -100, color=col_exc, label=r"$W^{A \rightarrow C} $")
    axes[1][i].plot(0, -100, color=col_inh,  label=r"$W^{B \rightarrow C} $")

    scale = max(np.abs(dend_error)) * 1.05
    axes[0][i].vlines(t_sim, -scale, scale, color="black", linestyle="--")
    axes[1][i].vlines(t_sim, -wmax, wmax, color="black", linestyle="--")

    axes[0][i].set_ylim(-scale, scale)
    axes[1][i].set_ylim(-(wmax+0.1), wmax+0.1)

    axes[1][i].set_title("Incoming synaptic weights")
    axes[1][i].set_xlabel("Simulation time [ms]")

    nest.ResetKernel()
axes[1][0].set_ylabel("weight [a.u.]")
axes[1][0].legend(loc='center left', bbox_to_anchor=[1, 0.5], fancybox=True)

curdir = os.path.dirname(os.path.realpath(__file__))
plt.savefig(os.path.join(curdir, "../../data/fig_exc_inh_split.png"))
# plt.show()
