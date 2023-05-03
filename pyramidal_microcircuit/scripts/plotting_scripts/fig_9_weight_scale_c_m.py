import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils

styles_c_m = {1: "-",
              10: ":",
              50: "--"}


colors_weight_scale = {1: "green",
                       5: "blue",
                       10: "red",
                       50: "orange"}


filter_window = 4
if __name__ == "__main__":

    plot_utils.setup_plt()

    args = sys.argv[1:]

    dirname = args[0]
    out_file = args[1]

    all_configs = sorted(os.listdir(dirname))
    fig, ax0 = plt.subplots(1, 1)

    orig_data_1 = []
    le_data_1 = []

    tau_eff = 1 / 0.19

    accuracies = {c_m: [] for c_m in styles_c_m.keys()}
    for config in all_configs:

        with open(os.path.join(dirname, config, "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join(dirname, config, "params.json")) as f:
            params = json.load(f)

        weight_scale = int(params["weight_scale"])
        c_m = float(params["C_m_api"])
        acc = progress["test_acc"]

        final_acc = np.mean([datapoint[1] for datapoint in acc[-10:]])  # average over last 10 accuracy readings
        orig_data_1.append((weight_scale, final_acc))

        times = [entry[0] for entry in acc]
        acc = [1-entry[1] for entry in acc]
        accuracies[c_m].append(final_acc)
        ax0.plot(times, utils.rolling_avg(acc, filter_window),
                 color=colors_weight_scale[weight_scale], linestyle=styles_c_m[c_m])
        # ax0.plot(times, acc,
        #  label=r"$t_{{pres}}={} \tau_{{eff}}$".format(round(t_pres)), color="orange", linestyle=linestyles[params["t_pres"]])

    # ax0.legend()
    le_data_1 = sorted(le_data_1)
    orig_data_1 = sorted(orig_data_1)

    le_data_1 = [[t, 1-acc] for [t, acc] in le_data_1]
    orig_data_1 = [[t, 1-acc] for [t, acc] in orig_data_1]
    correllation = [[k, np.mean(v)] for [k, v] in sorted(accuracies.items())]

    ax0.set_ylim(0, 1)

    ax0.set_xlabel("epoch")
    ax0.set_ylabel("test error")


    dummy_lines = [[ax0.plot((0, -1), color="black", linestyle=stl)[0], r"c_m = {}".format(c_m)]
                   for (c_m, stl) in styles_c_m.items()]
    legend2 = ax0.legend(*zip(*dummy_lines), loc=1)

    dummy_lines_2 = [[ax0.plot((0, -1), color=col)[0], r"weight scale = {}".format(w_s)]
                     for [w_s, col] in colors_weight_scale.items()]
    legend1 = ax0.legend(*zip(*dummy_lines_2), loc=4)
    # sax0.add_artist(legend1)
    ax0.add_artist(legend2)

    plt.savefig(out_file)
