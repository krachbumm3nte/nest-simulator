import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils

styles_weight_scale = {1: "solid",
                       5: "dashed",
                       10: "dotted",
                       50: "-."}


colors_c_m = {1: "green",
              5: "blue",
              15: "red"}


filter_window = 4
if __name__ == "__main__":

    plot_utils.setup_plt()

    args = sys.argv[1:]

    dirname = args[0]
    out_file = args[1]

    all_configs = sorted(os.listdir(dirname))
    fig, [ax0, ax1] = plt.subplots(2, 1)

    orig_data_1 = []
    le_data_1 = []

    tau_eff = 1 / 0.19

    accuracies = {c_m: [] for c_m in [1, 5, 15]}
    for config in all_configs:

        with open(os.path.join(dirname, config, "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join(dirname, config, "params.json")) as f:
            params = json.load(f)

        weight_scale = int(params["weight_scale"])
        c_m = int(params["c_m_api"])
        acc = progress["test_acc"]

        final_acc = np.mean([datapoint[1] for datapoint in acc[-10:]])  # average over last 10 accuracy readings
        orig_data_1.append((weight_scale, final_acc))

        times = [entry[0] for entry in acc]
        acc = [1-entry[1] for entry in acc]
        accuracies[c_m].append(final_acc)
        ax0.plot(times, utils.rolling_avg(acc, filter_window),
                 color=colors_c_m[c_m], linestyle=styles_weight_scale[weight_scale])
        # ax0.plot(times, acc,
        #  label=r"$t_{{pres}}={} \tau_{{eff}}$".format(round(t_pres)), color="orange", linestyle=linestyles[params["sim_time"]])

    # ax0.legend()
    le_data_1 = sorted(le_data_1)
    orig_data_1 = sorted(orig_data_1)

    le_data_1 = [[t, 1-acc] for [t, acc] in le_data_1]
    orig_data_1 = [[t, 1-acc] for [t, acc] in orig_data_1]
    correllation = [[k, np.mean(v)] for [k, v] in sorted(accuracies.items())]
    ax1.plot(correllation)

    ax0.set_ylim(0, 1)
    ax1.set_ylim(0, 1)

    ax0.set_xlabel("epoch")
    ax0.set_ylabel("test error")
    ax1.set_xlabel("Apical compartment capacitance")
    ax1.set_ylabel("test error")

    ax0.annotate("A", xy=(0.02, 0.985), xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=20)

    ax0.annotate("B", xy=(0.02, 0.485), xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=20)

    dummy_lines = [[ax1.plot((0, -1), color="black", linestyle=stl)[0], r"weight scale = {}".format(ws)]
                   for (ws, stl) in styles_weight_scale.items()]
    legend2 = ax0.legend(*zip(*dummy_lines), loc=1)

    dummy_lines_2 = [[ax1.plot((0, -1), color=col)[0], r"c_m = {}".format(c_m)] for [c_m, col] in colors_c_m.items()]
    legend1 = ax0.legend(*zip(*dummy_lines_2), loc=4)
    # sax0.add_artist(legend1)
    ax0.add_artist(legend2)

    plt.savefig(out_file)
