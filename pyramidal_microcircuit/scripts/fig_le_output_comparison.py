import numpy as np
from src.params import Params
import src.plot_utils as plot_utils
import matplotlib.pyplot as plt
import sys
import json
from src.networks.network_numpy import NumpyNetwork
from copy import deepcopy
p = Params()

ls = ["solid", "dashed", "dotted"]
if __name__ == "__main__":

    weights = "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/results/par_study_t_pres_le/bars_numpy/bars_le_tpres_5000_numpy/data/weights_1000.json"
    #weights = sys.argv[1]
    #out_file = sys.argv[2]
    plot_utils.setup_plt()
    p.eta = {
        'ip': [0.0, 0, 0],
        'pi': [0, 0, 0],
        'up': [0.0, 0.0, 0],
        'down': [0, 0, 0]
    }
    p.out_lag = 0
    p.sim_time = 50

    p.latent_equilibrium = False
    net_sac = NumpyNetwork(deepcopy(p))
    p.latent_equilibrium = True
    net_le = NumpyNetwork(deepcopy(p))

    with open(weights) as f:
        wgts = json.load(f)
    net_le.set_all_weights(wgts)
    net_sac.set_all_weights(wgts)

    stim_in, stim_out = net_le.get_training_data(1)
    stim_out = [np.zeros(p.dims[-1])]
    print("training...")
    net_sac.train_batch(stim_in, stim_out)
    net_le.train_batch(stim_in, stim_out)
    phi = net_le.phi
    print("plotting...")
    fig, axes = plt.subplots(2, sharex=True)
    for i, neuron_idx in enumerate([0]):
        for net, label, line_col in zip([net_le, net_sac], ["LE", "Sacramento"], ["blue", "orange"]):
            uh = net.U_h_record[:, neuron_idx]
            vbh = net.V_bh_record[:, neuron_idx]
            ui = net.U_i_record[:, neuron_idx]
            vbi = net.V_bi_record[:, neuron_idx]
            vah = net.V_ah_record[:, neuron_idx]
            uy = net.U_y_record
            vby = net.V_by_record[:, neuron_idx]

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
