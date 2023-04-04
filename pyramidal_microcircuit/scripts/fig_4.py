import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils

linestyles = {10: "solid",
              30: "dashed",
              100: "dotted"}

filter_window = 4
if __name__ == "__main__":

    plot_utils.setup_plt()

    args = sys.argv[1:]

    dirname = args[0]
    out_file = args[1]

    all_configs = sorted(os.listdir(dirname))
    fig, [ax0, ax1, ax2] = plt.subplots(3, 1)

    final_performance = []
    training_duration = []

    for config in all_configs:

        with open(os.path.join(dirname, config, "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join(dirname, config, "params.json")) as f:
            params = json.load(f)

        n_hidden = params["dims"][1]
        acc = np.array(progress["test_acc"])

        final_acc = np.mean(acc[-5:, 1])  # average over last 10 accuracy readings

        final_performance.append([n_hidden, final_acc])

        if not np.any(acc[:, 1] == 1.0):
            t_success = 10000
        else:
            t_success = np.where(acc[:, 1] < 1.0)[0]
            t_success = acc[t_success[-1], 0]

        training_duration.append([n_hidden, t_success])

        if n_hidden in [10, 30, 100]:
            times = [entry[0] for entry in acc]
            acc = [1-entry[1] for entry in acc]
            ax0.plot(times, utils.rolling_avg(acc, filter_window), label=r"$n_{{hidden}}={}$".format(
                n_hidden), color="orange", linestyle=linestyles[n_hidden])


    ax1.plot(*zip(*sorted(final_performance)))
    ax2.plot(*zip(*sorted(training_duration)))
    ax2.set_ylim(-5, 1000)
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("test loss")
    ax1.set_ylabel("training duration")
    ax1.set_xlabel("n hidden")
    ax0.legend()
    # plt.show()

    plt.savefig(out_file)
