import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.plot_utils as plot_utils
import src.utils as utils
from sklearn.metrics import mean_squared_error as mse
from src.networks.network_nest import NestNetwork
from src.params import Params

import nest

styles_c_m = {1: "-",
              10: ":",
              50: "--"}


colors_psi = {5: "green",
              10: "red",
              50: "blue"}

plot_utils.setup_plt()

test_samples = 16

filter_window = 4
if __name__ == "__main__":

    curdir = os.path.dirname(os.path.realpath(__file__))

    result_dir = os.path.join(curdir, "../../results/par_study_c_m_psi")
    out_file = os.path.join(curdir, "../../data/fig_c_m_psi.png")

    all_configs = sorted(os.listdir(result_dir))
    fig, [ax0, ax1] = plt.subplots(1, 2)

    error_all = {5: [], 10: [], 50: []}
    loss_all = {5: [], 10: [], 50: []}

    accuracies = {c_m: [] for c_m in styles_c_m.keys()}
    for config in all_configs:

        params = Params(os.path.join(result_dir, config, "params.json"))
        params.noise = False
        params.sigma = 0
        params.threads = 8
        params.eta = {
            "up": [0, 0],
            "down": [0, 0],
            "pi": [0, 0],
            "ip": [0, 0]
        }
        psi = params.psi
        c_m = params.C_m_api
        print(f"\n\nTesting for c_m={c_m}, psi={psi}")
        utils.setup_nest(params)

        net = NestNetwork(params)

        with open(os.path.join(result_dir, config, "weights.json")) as f:
            wgts = json.load(f)
        net.set_all_weights(wgts)

        with open(os.path.join(result_dir, config, "progress.json")) as f:
            progress = json.load(f)

        net_test_acc = []
        net_test_loss = []
        y_pred_total = []

        x_batch, y_batch = net.get_test_data(test_samples)
        net.disable_plasticity()
        for i, (x, y) in enumerate(zip(x_batch, y_batch)):
            net.set_input(x)
            net.simulate(net.t_pres, True)
            mm_data = pd.DataFrame.from_dict(net.mm.events)
            U_Y = [mm_data[mm_data["senders"] == out_id]["V_m.s"] for out_id in net.layers[-1].pyr.global_id]
            y_pred = np.mean(U_Y, axis=1)

            net_test_loss.append(mse(y, y_pred))
            net_test_acc.append(np.argmax(y) == np.argmax(y_pred))
            y_pred_total.append(y_pred)
            net.reset()

        final_loss = np.mean(net_test_loss)
        # we add a minor shift to all accuracies. This ensures that dots don't fully overlap in the figure
        final_acc = np.mean(net_test_acc) - 0.05 * final_loss
        error_all[psi].append((c_m, 1-final_acc))
        loss_all[psi].append((c_m, final_loss))

        nest.ResetKernel()

    # ax0.legend()

    for psi, data in error_all.items():
        data = sorted(data)
        ax0.scatter(np.arange(3), [i[1] for i in data], color=colors_psi[psi])

    for psi, data in loss_all.items():
        data = sorted(data)
        ax1.scatter(np.arange(3), [i[1] for i in data], color=colors_psi[psi])

    ax0.set_title("Test error")
    ax1.set_title("Test loss")
    ax0.set_xticks(np.arange(3))
    ax0.set_xticklabels([1, 10, 50])
    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels([1, 10, 50])
    ax0.set_ylim(bottom=-0.005)
    ax1.set_ylim(bottom=0)
    ax0.set_xlim(-0.5, 2.5)
    ax1.set_xlim(-0.5, 2.5)
    ax0.set_xlabel("$C_m^{api}$")
    ax1.set_xlabel("$C_m^{api}$")

    dummy_lines_2 = [[ax1.scatter(0, -1, color=col), r"$\psi = {}$".format(w_s)]
                     for [w_s, col] in sorted(colors_psi.items())]
    legend1 = ax1.legend(*zip(*dummy_lines_2), loc="best")
    ax1.add_artist(legend1)

    plt.savefig(out_file)
