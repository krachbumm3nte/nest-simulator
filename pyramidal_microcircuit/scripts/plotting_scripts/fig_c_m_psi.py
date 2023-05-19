import json
import os

import matplotlib.pyplot as plt
import numpy as np

styles_c_m = {1: "-",
              10: ":",
              50: "--"}


colors_psi = {100: "green",
              10: "red",
              50: "blue"}


filter_window = 4
if __name__ == "__main__":

    curdir = os.path.dirname(os.path.realpath(__file__))

    dirname = os.path.join(curdir, "../../results/par_study_c_m_psi")
    out_file = os.path.join(curdir, "../../data/fig_c_m_psi.png")

    all_configs = sorted(os.listdir(dirname))
    fig, [ax0, ax1] = plt.subplots(1, 2)

    n_epochs = 30

    error_all = {10: [], 50: [], 100: []}
    loss_all = {10: [], 50: [], 100: []}

    accuracies = {c_m: [] for c_m in styles_c_m.keys()}
    for config in all_configs:

        with open(os.path.join(dirname, config, "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join(dirname, config, "params.json")) as f:
            params = json.load(f)

        psi = int(params["psi"])
        c_m = float(params["C_m_api"])
        print(f"processing c_m {c_m}, psi {psi}")
        acc = progress["test_acc"]

        final_acc = np.mean([datapoint[1] for datapoint in acc[-n_epochs:]])

        loss = progress["test_loss"]
        final_loss = np.mean([datapoint[1] for datapoint in loss[-n_epochs:]])  # average over last 10 accuracy readings

        error_all[psi].append((c_m, 1-final_acc))
        loss_all[psi].append((c_m, final_loss))

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
    ax0.set_ylim(0, 0.5)
    ax1.set_ylim(0, 0.25)
    ax0.set_xlim(-0.5, 2.5)
    ax1.set_xlim(-0.5, 2.5)
    ax0.set_xlabel("$C_m^{api}$")
    ax1.set_xlabel("$C_m^{api}$")

    dummy_lines_2 = [[ax0.scatter(0, -1, color=col), r"$\psi = {}$".format(w_s)]
                     for [w_s, col] in sorted(colors_psi.items())]
    legend1 = ax0.legend(*zip(*dummy_lines_2), loc=0)
    ax0.add_artist(legend1)

    plt.savefig(out_file)
