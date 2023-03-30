import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils
from sklearn.metrics import mean_squared_error as mse


filter_window = 4
if __name__ == "__main__":

    plot_utils.setup_plt()

    args = sys.argv[1:]

    dirname = args[0]
    out_file = args[1]

    all_configs = sorted(os.listdir(dirname))
    fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2, sharex=True)

    final_performance = []
    training_duration = []

    for config in all_configs:

        network_type = config.split("_")[-2]
        col = plot_utils.colors[network_type]

        with open(os.path.join(dirname, config, "progress.json")) as f:
            progress = json.load(f)

        apical_error = np.array(progress["apical_error"])
        intnt_error = np.array(progress["intn_error"])

        ax0.plot(apical_error[:, 0], utils.rolling_avg(apical_error[:, 1], 100), color=col, label=network_type)
        ax1.plot(intnt_error[:, 0], utils.rolling_avg(intnt_error[:, 1], 100), color=col, label=network_type)

        ff_error = []
        fb_error = []
        curdir = os.path.join(dirname, config)
        for file in os.listdir(os.path.join(curdir, "data"), ):
            with open(os.path.join(curdir, "data", file), "r") as f:
                weights = json.load(f)

            epoch = int(file.split('.')[0].split("_")[-1])

            WHI = np.array(weights[-2]["pi"])
            WHY = np.array(weights[-2]["down"])
            WIH = np.array(weights[-2]["ip"])
            WYH = np.array(weights[-1]["up"])

            ff_error.append([epoch, mse(WYH.flatten(), WIH.flatten())])
            fb_error.append([epoch, mse(WHY.flatten(), -WHI.flatten())])

        ax2.plot(*zip(*sorted(ff_error)), color=col, label=network_type)
        ax3.plot(*zip(*sorted(fb_error)), color=col, label=network_type)

    ax2.set_xlabel(epoch)
    ax2.set_xlabel(epoch)
    ax0.set_title("apical error")
    ax1.set_title("interneuron error")
    ax2.set_title("feedforward weight error")
    ax3.set_title("feedback weight error")

    ax0.legend()

    # plt.show()

    plt.savefig(out_file)
