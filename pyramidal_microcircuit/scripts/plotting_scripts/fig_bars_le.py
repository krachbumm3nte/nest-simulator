# -*- coding: utf-8 -*-
#
# fig_bars_le.py
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

    for config in configs_orig:

        with open(os.path.join(dirname, config, "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join(dirname, config, "params.json")) as f:
            params = json.load(f)

        t_pres = params["t_pres"]
        acc = progress["test_acc"]

        final_acc = np.mean([datapoint[1] for datapoint in acc[-10:]])  # average over last 10 accuracy readings
        orig_data_1.append((t_pres, final_acc))

        if params["t_pres"] in [500, 50, 5]:
            times = [entry[0] for entry in acc]
            acc = [1-entry[1] for entry in acc]
            ax0.plot(times, utils.rolling_avg(acc, filter_window), label=r"$t_{{pres}}={}ms$".format(round(t_pres)),
                     color="orange", linestyle=linestyles[params["t_pres"]])

    for config in configs_le:

        with open(os.path.join(dirname, config, "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join(dirname, config, "params.json")) as f:
            params = json.load(f)

        t_pres = params["t_pres"]
        acc = progress["test_acc"]

        final_acc = np.mean([datapoint[1] for datapoint in acc[-10:]])  # average over last 10 accuracy readings
        le_data_1.append((t_pres, final_acc))

        if params["t_pres"] in [500, 50, 5]:
            times = [entry[0] for entry in acc]
            acc = [1-entry[1] for entry in acc]
            ax0.plot(times, utils.rolling_avg(acc, filter_window), label=r"$t_{{pres}}={}ms$, le".format(round(t_pres)),
                     color="blue", linestyle=linestyles[params["t_pres"]])

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
    ax1.set_xlabel(r'$t_{pres}  \left[ ms \right]$')
    ax1.set_ylabel("test error")

    lines = ax0.get_lines()

    ax0.annotate("A", xy=(0.02, 0.985), xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=20)

    ax0.annotate("B", xy=(0.02, 0.485), xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=20)

    legend1 = ax0.legend(lines[:3], [r"$t_{{pres}}={}ms$".format(t) for t in [500, 50, 5]], loc=1)
    legend2 = ax0.legend(lines[2::3], ["Sacramento", "Latent Equilibrium"], loc=4)
    ax0.add_artist(legend1)

    # plt.show()

    plt.savefig(out_file)
