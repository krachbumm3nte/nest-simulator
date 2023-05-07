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

    curdir = os.path.dirname(os.path.realpath(__file__))

    result_dir = os.path.join(curdir, "../../results/par_study_function_approximator")
    out_file = os.path.join(curdir, "../../data/fig_function_approximator.png")

    dirnames = os.listdir(result_dir)
    acc = []
    loss = []
    fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2, sharex=True)

    all_configs = sorted([name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))])

    n_samples = 2
    for config in all_configs:

        params = Params(os.path.join(result_dir, config, "params.json"))

        n = params.dims[1]

        with open(os.path.join(result_dir, config, "progress.json")) as f:
            progress = json.load(f)

        acc.append([n, np.mean([i[1] for i in progress["test_acc"][-n_samples:]])])
        loss.append([n, np.mean([i[1] for i in progress["test_loss"][-n_samples:]])])

    print(acc)
    print(loss)
    ax0.plot(*zip(*sorted(acc)))
    ax1.plot(*zip(*sorted(loss)))

    ax2.set_xlabel(r"Dropout (\%)")
    ax3.set_xlabel(r"Dropout (\%)")
    ax0.set_title("Apical error")
    ax1.set_title("Interneuron error")
    ax2.set_title("Feedforward weight error")
    ax3.set_title("Feedback weight error")

    ax0.set_ylim(bottom=0)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)

    # plt.show()

    plt.savefig(out_file)
