# -*- coding: utf-8 -*-
#
# fig_reset_strategies.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

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
    fig, [ax0, ax1] = plt.subplots(1, 2,  sharex=True, figsize=[8, 3])

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

    ax0.set_xlim(0, 600)
    ax1.set_xlim(0, 600)

    ax0.set_ylim(-0.01, 0.6)
    ax1.set_ylim(-0.005, 0.25)

    ax0.set_title("Test error")
    ax1.set_title("Test loss")

    ax0.set_xlabel("Epoch")
    ax1.set_xlabel("Epoch")

    ax0.legend()

    plt.savefig(out_file)
