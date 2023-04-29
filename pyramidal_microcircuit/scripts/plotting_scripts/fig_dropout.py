import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils
from sklearn.metrics import mean_squared_error as mse

from src.params import Params

filter_window = 4
if __name__ == "__main__":

    plot_utils.setup_plt()

    args = sys.argv[1:]

    directory = args[0]
    out_file = args[1]
    all_configs = sorted([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])
    fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2, sharex=True)


    apical_errors = []
    intn_errors= []
    ff_errors = []
    fb_errors = []

    for config in all_configs:

        params = Params(os.path.join(directory, config, "params.json"))

        p = 100*(1-params.p_conn)

        with open(os.path.join(directory, config, "progress.json")) as f:
            progress = json.load(f)


        apical_error = np.array(progress["apical_error"])
        intn_error = np.array(progress["intn_error"])
        print()
        apical_errors.append([p, np.mean(apical_error[-50:, 1])])
        intn_errors.append([p, np.mean(intn_error[-50:, 1])])

        with open(os.path.join(directory, config, "data", "weights_2000.json")) as f:
            weights = json.load(f)


        WHI = np.array(weights[-2]["pi"])
        WHY = np.array(weights[-2]["down"])
        WIH = np.array(weights[-2]["ip"])
        WYH = np.array(weights[-1]["up"])

        for w_arr in [WHI, WHY, WIH, WYH]:
            w_arr[np.isnan(w_arr)] = 0

        ff_errors.append([p, mse(WYH.flatten(), WIH.flatten())])
        fb_errors.append([p, mse(WHY.flatten(), -WHI.flatten())])

    print(apical_errors)
    ax0.plot(*zip(*sorted(apical_errors)))
    ax1.plot(*zip(*sorted(intn_errors)))

    ax2.plot(*zip(*sorted(ff_errors)))
    ax3.plot(*zip(*sorted(fb_errors)))

    ax2.set_xlabel("Dropout (%)")
    ax3.set_xlabel("Dropout (%)")
    ax0.set_title("Apical error")
    ax1.set_title("Interneuron error")
    ax2.set_title("Feedforward weight error")
    ax3.set_title("Feedback weight error")

    # ax0.set_ylim(bottom=0, top=0.005)
    # ax1.set_ylim(bottom=0, top=0.00001)
    # ax2.set_ylim(bottom=0, top=0.15)
    # ax3.set_ylim(bottom=0)

    ax1.legend()

    # plt.show()

    plt.savefig(out_file)
