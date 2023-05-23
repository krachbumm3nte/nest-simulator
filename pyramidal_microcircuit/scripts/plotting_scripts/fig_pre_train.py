import json
import os

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils

filter_window = 4
test_samples = 250

conf_names = {"bars_default_snest": "Self-predicting weights",
              "bars_no_selfpred_snest": "Random weights",
              "bars_no_selfpred_soft_reset_snest": "Random weights + soft reset"}

if __name__ == "__main__":

    plot_utils.setup_plt()

    curdir = os.path.dirname(os.path.realpath(__file__))

    result_dir = os.path.join(curdir, "../../results/par_study_pre_train")
    out_file = os.path.join(curdir, "../../data/fig_pre_train.png")

    dirnames = os.listdir(result_dir)
    test_error = []
    test_loss = []
    train_loss = []
    r2_scores = []
    fig, [ax0, ax1] = plt.subplots(1, 2, sharex=True, figsize=[8, 3])

    all_configs = sorted([name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))])

    n_samples = 3
    for config in all_configs:

        with open(os.path.join(result_dir, config, "progress.json")) as f:
            progress = json.load(f)

        test_acc = np.array(sorted(progress["test_acc"]))
        test_loss = np.array(sorted(progress["test_loss"]))
        test_acc[:, 1] = 1 - test_acc[:, 1]
        ax0.plot(test_acc[:, 0], utils.rolling_avg(test_acc[:, 1], 2), label=conf_names[config])
        ax1.plot(test_loss[:, 0], utils.rolling_avg(test_loss[:, 1], 10), label=conf_names[config])
    ax0.set_xlim(0, 400)
    ax1.set_xlim(0, 400)

    ax0.set_ylim(-0.01, 0.6)
    ax1.set_ylim(-0.005, 0.25)

    ax0.set_title("Test error")
    ax1.set_title("Test loss")

    ax0.set_xlabel("Epoch")
    ax1.set_xlabel("Epoch")

    ax0.legend()

    # plt.show()
    plt.savefig(out_file)
