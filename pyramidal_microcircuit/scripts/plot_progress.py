# -*- coding: utf-8 -*-
#
# plot_progress.py
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
import src.plot_utils as plot_utils
from src.params import Params

from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork


if __name__ == "__main__":

    plot_utils.setup_plt()

    args = sys.argv[1:]

    dirname = args[0]
    out_file = args[1]

    fig, [ax0, ax1, ax2] = plt.subplots(3, 1)

    final_performance = []
    training_duration = []

    p = Params(os.path.join(dirname, "params.json"))
    p.p_conn = 1
    with open(os.path.join(dirname, "progress.json")) as f:
        progress = json.load(f)

    if p.network_type == "numpy":
        net = NumpyNetwork(p)
    else:
        net = NestNetwork(p)

    datadir = os.path.join(dirname, "data")

    for filename in sorted(os.listdir(datadir)):
        with open(os.path.join(datadir, filename)) as f:
            weights = json.load(f)
            fb_error, ff_error, WHI, WHY, WIH, WYH = plot_utils.calculate_weight_errors(weights)
            epoch = int(filename.split(".")[0].split("_")[-1])
            net.fb_error.append([epoch, fb_error])
            net.ff_error.append([epoch, ff_error])

    net.fb_error = sorted(net.fb_error)
    net.ff_error = sorted(net.ff_error)

    net.test_acc = progress["test_acc"]
    net.test_loss = progress["test_loss"]
    net.train_loss = progress["train_loss"]
    # net.ff_error = progress["ff_error"]
    # net.fb_error = progress["fb_error"]
    net.intn_error = progress["intn_error"]
    net.apical_error = progress["apical_error"]
    net.epoch = progress["epochs_completed"]
    print(net.epoch)
    plot_utils.plot_training_progress(net.epoch, net, out_file)
    # plot_utils.plot_pre_training(net.epoch, net, out_file)
