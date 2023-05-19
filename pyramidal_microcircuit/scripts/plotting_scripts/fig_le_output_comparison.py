import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.plot_utils as plot_utils
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.params import Params

p = Params()
utils.setup_nest(p)

ls = ["solid", "dashed", "dotted"]
if __name__ == "__main__":

    weights = "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/results/par_study_t_pres_le/bars_snest/bars_le_tpres_1000_snest/data/weights_1000.json"
    plot_utils.setup_plt()
    p.eta = {
        'ip': [0.0, 0, 0],
        'pi': [0, 0, 0],
        'up': [0.0, 0.0, 0],
        'down': [0, 0, 0]
    }

    p.out_lag = 0
    p.t_pres = 50
    p.store_errors = True
    p.spiking = False
    p.C_m_api = 50
    p.latent_equilibrium = False
    net_sac = NestNetwork(deepcopy(p))
    p.latent_equilibrium = True
    net_le = NestNetwork(deepcopy(p))

    net_le.redefine_connections()
    net_sac.redefine_connections()

    with open(weights) as f:
        wgts = json.load(f)

    print(net_le.dims, net_sac.dims)
    net_le.set_all_weights(wgts)
    net_sac.set_all_weights(wgts)

    stim_in, stim_out = net_le.get_training_data(1)
    stim_in = [stim_in[0]]
    #stim_out = [np.zeros(p.dims[-1])]
    stim_out = [stim_out[0]]
    print(stim_in, stim_out)
    print("training...")
    net_sac.train_batch(stim_in, stim_out)
    net_le.train_batch(stim_in, stim_out)
    phi = net_le.phi
    print("plotting...")
    fig, axes = plt.subplots(2, 2, sharex=True)
    for i, neuron_idx in enumerate([0]):
        for net, label, line_col in zip([net_le, net_sac], ["LE", "Sacramento"], ["blue", "orange"]):
            df = pd.DataFrame.from_dict(net.mm.get("events"))
            uh = utils.get_mm_data(df, net.layers[-2].pyr[neuron_idx], "V_m.s")
            vbh = utils.get_mm_data(df, net.layers[-2].pyr[neuron_idx], "V_m.b")
            vah = utils.get_mm_data(df, net.layers[-2].pyr[neuron_idx], "V_m.a_lat")
            ui = utils.get_mm_data(df, net.layers[-2].intn[neuron_idx], "V_m.s")
            vbi = utils.get_mm_data(df, net.layers[-2].intn[neuron_idx], "V_m.b")
            uy = utils.get_mm_data(df, net.layers[-1].pyr, "V_m.s")
            vby = utils.get_mm_data(df, net.layers[-1].pyr[neuron_idx], "V_m.b")

            basal_pred_y = p.g_d * vby / (p.g_l + p.g_d)

            basal_pred_intn = p.g_d * vbi / (p.g_l + p.g_d)

            w_down = np.array(wgts[-2]["down"])

            vah_dist = np.array([w_down @ phi(uy_t) for uy_t in uy])

            axes[0][0].plot(phi(uy[:, neuron_idx]) - phi(basal_pred_y),
                            label=label, linestyle=ls[i], color=line_col)

            axes[0][1].plot(phi(ui) - phi(basal_pred_intn), label=label, linestyle=ls[i], color=line_col)

            axes[1][0].plot(-vah, label=label, linestyle=ls[i], color=line_col)

            axes[1][1].plot(phi(uh) - vah_dist[:, neuron_idx], label=label, linestyle=ls[i], color=line_col)

    axes[0][0].legend()
    axes[0][0].set_title(r"\textbf{A:} $ \phi(u_N^{P}) - \phi(\hat{v}_N^{bas}) $")
    axes[0][1].set_title(r"\textbf{B:} $ \phi(u_{N-1}^{I}) - \phi(\hat{v}_{N-1}^{dend}) $")
    axes[1][0].set_title(r"\textbf{C:} $- v_{N-1}^{api}$")
    axes[1][1].set_title(r"\textbf{D:} $\phi(u_{N-1}^{P}) - \phi(w_{N-1}^{down} r_{N}^P) $")

    axes[1][0].set_xlabel(r"$t_{pres} \left[ ms \right]$")
    axes[1][1].set_xlabel(r"$t_{pres} \left[ ms \right]$")

    # axes[0][0].set_ylim(bottom=-0.005)
    # axes[0][1].set_ylim(bottom=-0.005)
    # axes[1][0].set_ylim(bottom=-0.0005)
    # axes[1][1].set_ylim(bottom=-0.005)
    plt.show()
    # plt.savefig(out_file)
