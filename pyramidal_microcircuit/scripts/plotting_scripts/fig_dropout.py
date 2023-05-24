# -*- coding: utf-8 -*-
#
# fig_dropout.py
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

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
from sklearn.metrics import mean_squared_error as mse
from src.params import Params

filter_window = 4
if __name__ == "__main__":

    plot_utils.setup_plt()

    curdir = os.path.dirname(os.path.realpath(__file__))

    result_dir = os.path.join(curdir, "../../results/par_study_dropout")
    out_file = os.path.join(curdir, "../../data/fig_dropout.png")

    dirnames = os.listdir(result_dir)
    apical_errors = {}
    intn_errors = {}
    ff_errors = {}
    fb_errors = {}
    fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2, sharex=True)

    for d in dirnames:
        directory = os.path.join(result_dir, d)
        all_configs = sorted([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

        for config in all_configs:

            params = Params(os.path.join(directory, config, "params.json"))

            p = 100*(1-params.p_conn)  # neuron dropout percentage

            with open(os.path.join(directory, config, "progress.json")) as f:
                progress = json.load(f)

            apical_error = np.array(progress["apical_error"])
            intn_error = np.array(progress["intn_error"])

            n_avg = 1500
            if p not in apical_errors:
                apical_errors[p] = [np.mean(apical_error[-n_avg:, 1])]
            else:
                apical_errors[p].append(np.mean(apical_error[-n_avg:, 1]))

            if p not in intn_errors:
                intn_errors[p] = [np.mean(intn_error[-n_avg:, 1])]
            else:
                intn_errors[p].append(np.mean(intn_error[-n_avg:, 1]))

            WHI = []
            WHY = []
            WIH = []
            WYH = []

            datadir = os.path.join(directory, config, "data")
            for weight_file in sorted(os.listdir(datadir)[-5:]):
                with open(os.path.join(datadir, weight_file)) as f:
                    weights = json.load(f)

                    WHI.append(weights[-2]["pi"])
                    WHY.append(weights[-2]["down"])
                    WIH.append(weights[-2]["ip"])
                    WYH.append(weights[-1]["up"])

            WHI = np.array(WHI)
            WHY = np.array(WHY)
            WIH = np.array(WIH)
            WYH = np.array(WYH)

            WHI = np.mean(WHI, axis=0)
            WHY = np.mean(WHY, axis=0)
            WIH = np.mean(WIH, axis=0)
            WYH = np.mean(WYH, axis=0)

            absent_whi = np.isnan(WHI)
            print(np.mean(np.abs(WHY)))

            absent_wih = np.isnan(WIH)
            print(np.mean(np.abs(WYH)))
            print()

            for w_arr in [WHI, WHY, WIH, WYH]:
                w_arr[np.isnan(w_arr)] = 0

            if p not in ff_errors:
                ff_errors[p] = [mse(WYH.flatten(), WIH.flatten())]
            else:
                ff_errors[p].append(mse(WYH.flatten(), WIH.flatten()))

            if p not in fb_errors:
                fb_errors[p] = [mse(WHY.flatten(), -WHI.flatten())]
            else:
                fb_errors[p].append(mse(WHY.flatten(), -WHI.flatten()))

    print(apical_errors)

    intn_errors = [[k, np.mean(v)] for (k, v) in intn_errors.items()]
    apical_errors = [[k, np.mean(v)] for (k, v) in apical_errors.items()]
    fb_errors = [[k, np.mean(v)] for (k, v) in fb_errors.items()]
    ff_errors = [[k, np.mean(v)] for (k, v) in ff_errors.items()]

    print(apical_errors)
    print(ff_errors)
    ax0.plot(*zip(*sorted(apical_errors)))
    ax1.plot(*zip(*sorted(intn_errors)))

    ax2.plot(*zip(*sorted(ff_errors)))
    ax3.plot(*zip(*sorted(fb_errors)))

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
