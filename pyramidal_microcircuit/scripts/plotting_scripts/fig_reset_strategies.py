import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils

colors_reset_type = {0: "red",
                     1: "blue",
                     2: "green"}


reset_keys = {
    0: "no reset",
    1: "soft reset",
    2: "hard reset"
}


filter_window = 4
if __name__ == "__main__":

    plot_utils.setup_plt()

    curdir = os.path.dirname(os.path.realpath(__file__))

    directory = os.path.join(curdir, "../../results/par_study_reset_strategies")
    out_file = os.path.join(curdir, "../../data/fig_reset_strategies.png")

    all_configs = sorted(os.listdir(directory))
    fig, [ax0, ax1] = plt.subplots(2, 1)

    orig_data_1 = []
    le_data_1 = []

    tau_eff = 1 / 0.19

    accuracies = {c_m: [] for c_m in [1, 2, 5, 10]}
    for config in all_configs:

        with open(os.path.join(directory, config, "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join(directory, config, "params.json")) as f:
            params = json.load(f)

        psi = int(params["psi"])
        reset = int(params["reset"])
        acc = np.array(progress["test_acc"])

        final_acc = np.mean([acc[-10:, 1]])  # average over last 10 accuracy readings
        orig_data_1.append((psi, final_acc))

        times = [entry[0] for entry in acc]
        acc = [1-entry[1] for entry in acc]

        loss = [entry[1] for entry in progress["test_loss"]]
        label = reset_keys[reset]

        ax0.plot(times, utils.rolling_avg(acc, filter_window), label=label)
        ax1.plot(times, utils.rolling_avg(loss, filter_window), label=label)


    le_data_1 = sorted(le_data_1)
    orig_data_1 = sorted(orig_data_1)

    le_data_1 = [[t, 1-acc] for [t, acc] in le_data_1]
    orig_data_1 = [[t, 1-acc] for [t, acc] in orig_data_1]

    ax0.set_ylim(0, 1)
    ax1.set_ylim(0, 1)

    ax0.set_ylabel("Test error")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test loss")

    ax0.annotate("A", xy=(0.02, 0.985), xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=20)

    ax0.annotate("B", xy=(0.02, 0.485), xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=20)
    ax0.legend()
    plt.savefig(out_file)
