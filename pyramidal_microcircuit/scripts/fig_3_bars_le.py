import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils

linestyles = {5: "solid",
              50: "dashed",
              500: "dotted"}

filter_window = 4
if __name__ == "__main__":

    plot_utils.setup_plt()

    args = sys.argv[1:]

    dirname = args[0]
    network_type = args[1]
    out_file = args[2]

    all_configs = sorted(os.listdir(dirname))
    configs_le = [name for name in all_configs if re.findall(f".+le_.+_{network_type}", name)]
    configs_orig = [name for name in all_configs if re.findall(f".+orig_.+_{network_type}", name)]
    print(all_configs)
    fig, [ax0, ax1] = plt.subplots(2, 1)

    orig_data_1 = []
    le_data_1 = []

    tau_eff = 1 / 0.19

    for config in configs_orig:

        with open(os.path.join(dirname, config, "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join(dirname, config, "params.json")) as f:
            params = json.load(f)

        t_pres = params["sim_time"] / tau_eff
        acc = progress["test_acc"]

        final_acc = np.mean([datapoint[1] for datapoint in acc[-10:]])  # average over last 10 accuracy readings
        orig_data_1.append((t_pres, final_acc))

        if params["sim_time"] in [500, 50, 5]:
            times = [entry[0] for entry in acc]
            acc = [1-entry[1] for entry in acc]
            ax0.plot(times, utils.rolling_avg(acc, filter_window),
                     label=r"$t_{{pres}}={} \tau_{{eff}}$".format(round(t_pres)), color="orange", linestyle=linestyles[params["sim_time"]])
            # ax0.plot(times, acc,
            #  label=r"$t_{{pres}}={} \tau_{{eff}}$".format(round(t_pres)), color="orange", linestyle=linestyles[params["sim_time"]])

    for config in configs_le:

        with open(os.path.join(dirname, config, "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join(dirname, config, "params.json")) as f:
            params = json.load(f)

        t_pres = params["sim_time"] / tau_eff
        acc = progress["test_acc"]

        final_acc = np.mean([datapoint[1] for datapoint in acc[-10:]])  # average over last 10 accuracy readings
        le_data_1.append((t_pres, final_acc))

        if params["sim_time"] in [500, 50, 5]:
            times = [entry[0] for entry in acc]
            acc = [1-entry[1] for entry in acc]
            ax0.plot(times, utils.rolling_avg(acc, filter_window),
                     label=r"$t_{{pres}}={} \tau_{{eff}}$, le".format(round(t_pres)), color="blue", linestyle=linestyles[params["sim_time"]])

    # ax0.legend()
    le_data_1 = sorted(le_data_1)
    orig_data_1 = sorted(orig_data_1)

    le_data_1 = [[t, 1-acc] for [t, acc] in le_data_1]
    orig_data_1 = [[t, 1-acc] for [t, acc] in orig_data_1]
    ax1.set_xscale("log")
    ax1.plot(*zip(*sorted(le_data_1)), color="blue", label="with le")
    ax1.plot(*zip(*sorted(orig_data_1)), color="orange", label="Sacramento")

    ax0.set_ylim(0, 1)
    ax0.set_xlim(0, 1000)
    ax1.set_ylim(0, 1)

    ax0.set_xlabel("epoch")
    ax0.set_ylabel("test error")
    ax1.set_xlabel(r'$t_{pres}  \left[ \tau_{eff} \right]$')
    ax1.set_ylabel("test error")

    lines = ax0.get_lines()

    ax0.annotate("A", xy=(0.02, 0.985), xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=20)
    
    ax0.annotate("B", xy=(0.02, 0.485), xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=20)

    legend1 = ax0.legend(lines[:3], [r"$t_{{pres}}={} \tau_{{eff}}$".format(t) for t in [100, 10, 1]], loc=1)
    legend2 = ax0.legend(lines[2::3], ["Sacramento", "Latent Equilibrium"], loc=4)
    ax0.add_artist(legend1)

    # plt.show()

    plt.savefig(out_file)
