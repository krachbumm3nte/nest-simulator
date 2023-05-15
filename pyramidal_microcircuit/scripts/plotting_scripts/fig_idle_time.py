import json
import os
import re
import sys
import nest
import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils
from sklearn.metrics import mean_squared_error as mse
from src.networks.network_nest import NestNetwork
from src.params import Params
import pandas as pd

filter_window = 4
test_samples = 250

conf_names = {"bars_le_default_snest": "Default",
              "bars_le_target_delay_snest": "Delayed target",
              "bars_le_target_delay_soft_reset_snest": "Delayed target + soft reset"}

if __name__ == "__main__":

    plot_utils.setup_plt()

    curdir = os.path.dirname(os.path.realpath(__file__))

    result_dir = os.path.join(curdir, "../../results/par_study_idle_time")
    out_file = os.path.join(curdir, "../../data/fig_idle_time.png")

    dirnames = os.listdir(result_dir)
    test_error = []
    test_loss = []
    train_loss = []
    r2_scores = []
    fig, [ax0, ax1] = plt.subplots(1, 2, sharex=True)

    all_configs = sorted([name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))])

    n_samples = 3
    for config in all_configs:

        with open(os.path.join(result_dir, config, "progress.json")) as f:
            progress = json.load(f)

        test_acc = np.array(sorted(progress["test_acc"]))
        test_loss = np.array(sorted(progress["test_loss"]))

        ax0.plot(test_acc[:, 0], utils.rolling_avg(test_acc[:, 1], 3), label=conf_names[config])
        ax1.plot(test_loss[:, 0], utils.rolling_avg(test_loss[:, 1], 10), label=conf_names[config])
    ax0.set_xlim(0, 500)
    ax1.set_xlim(0, 500)

    ax0.set_ylim(0.4, 1.05)

    ax0.set_title("Accuracy")
    ax1.set_title("Loss")

    ax0.set_xlabel("Epoch")
    ax1.set_xlabel("Epoch")

    ax0.legend()

    # plt.show()
    plt.savefig(out_file)
