# -*- coding: utf-8 -*-
#
# plot_utils.py
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


import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils
from sklearn.metrics import mean_squared_error as mse

ff_error = []
fb_error = []


colors = {
    "numpy": "orange",
    "rnest": "green",
    "snest": "blue",
}


def setup_plt():
    """Sets pyplot params for all plots
    """
    plt.rcParams['savefig.dpi'] = 500
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['text.usetex'] = True
    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 22}

    # plt.rcParams['font'] = font


def calculate_weight_errors(weights, layer_offset=0):
    # Synaptic weights
    # notice that all weights are scaled up again to ensure that derived metrics are comparible between simulations
    WHI = np.array(weights[-(layer_offset + 2)]["pi"])
    WHY = np.array(weights[-(layer_offset + 2)]["down"])
    WIH = np.array(weights[-(layer_offset + 2)]["ip"])
    WYH = np.array(weights[-(layer_offset + 1)]["up"])

    # handles neuron dropout
    WHI[np.isnan(WHI)] = 0
    WHY[np.isnan(WHY)] = 0
    WIH[np.isnan(WIH)] = 0
    WYH[np.isnan(WYH)] = 0

    fb_error = mse(WHY.flatten(), -WHI.flatten())

    ff_error = mse(WYH.flatten(), WIH.flatten())

    return fb_error, ff_error, WHI, WHY, WIH, WYH


def plot_pre_training(epoch, net, out_file):
    """Generates a plot specifically designed to record progress on the pre-training
    towards the self-predicting state.

    Arguments:
        epoch -- number of training epochs passed
        net -- instance of networks.Network
        out_file -- file to write plot to
    """
    print(f"Ep {epoch}: generating plot...", end="")

    cmap_2 = plt.cm.get_cmap('hsv', net.dims[2]+1)

    fig, axes = plt.subplots(3, 2, constrained_layout=True)
    [ax0, ax1, ax2, ax3, ax4, ax5] = axes.flatten()

    fb_error, ff_error, WHI, WHY, WIH, WYH = calculate_weight_errors(net.get_weight_dict())
    ax0.plot(*zip(*net.fb_error))
    ax1.plot(*zip(*net.ff_error))

    # plot synaptic weights
    for i in range(net.dims[-1]):
        col = cmap_2(i)
        for j in range(net.dims[-2]):
            ax2.plot(j, -WHY[j, i], ".", color=col, label=f"to {i}")
            ax2.plot(j, WHI[j, i], "x", color=col, label=f"from {i}")

    for i in range(net.dims[-2]):
        for j in range(net.dims[-1]):
            col = cmap_2(j)
            ax3.plot(i, WYH[j, i], ".", color=col, label=f"to {i}")
            ax3.plot(i, WIH[j, i], "x", color=col, label=f"from {i}")

    apical_error = np.array(net.apical_error)
    ax4.plot(apical_error[:, 0] * net.train_samples, utils.rolling_avg(apical_error[:, 1], 5))
    intn_error = np.array(net.intn_error)
    ax5.plot(intn_error[:, 0] * net.train_samples, utils.rolling_avg(intn_error[:, 1], 5))

    ax0.set_title("Feedback error")
    ax1.set_title("Feedforward error")
    ax2.set_title("Feedback weights")
    ax3.set_title("Feedforward weights")
    ax4.set_title("Apical error")
    ax5.set_title("Interneuron error")

    ax0.set_ylim(bottom=0)
    ax1.set_ylim(bottom=0)
    ax4.set_ylim(bottom=0)
    ax5.set_ylim(bottom=0)

    plt.savefig(out_file)
    plt.close()
    print(f"done.\n")


def plot_training_progress(epoch, net, out_file):
    """Generates a plot for general training progress.

    Arguments:
        epoch -- number of training epochs passed
        net -- instance of networks.Network
        out_file -- file to write plot to
    """
    print(f"Ep {epoch}: generating plot...", end="")

    cmap_1 = plt.cm.get_cmap('hsv', net.dims[1]+1)
    cmap_2 = plt.cm.get_cmap('hsv', net.dims[2]+1)

    fig, axes = plt.subplots(3, 2, constrained_layout=True)
    [ax0, ax1, ax2, ax3, ax4, ax5] = axes.flatten()

    # Synaptic weights
    # notice that all weights are scaled up again to ensure that derived metrics are comparible between simulations

    ax0.plot(*zip(*net.train_loss))
    ax1.plot(*zip(*net.ff_error))

    if sum(net.dims) < 100:
        fb_, ff_, WHI, WHY, WIH, WYH = calculate_weight_errors(net.get_weight_dict())

        # plot synaptic weights
        for i in range(net.dims[-1]):
            col = cmap_2(i)
            for j in range(net.dims[-2]):
                ax2.plot(j, -WHY[j, i], ".", color=col, label=f"to {i}")
                ax2.plot(j, WHI[j, i], "x", color=col, label=f"from {i}")

        for i in range(net.dims[-2]):
            for j in range(net.dims[-1]):
                col = cmap_2(j)
                ax3.plot(i, WYH[j, i], ".", color=col, label=f"to {i}")
                ax3.plot(i, WIH[j, i], "x", color=col, label=f"from {i}")

        if len(net.dims) > 3:
            fb_, ff_, WHI, WHY, WIH, WYH = calculate_weight_errors(net.get_weight_dict(), 1)

            for i in range(net.dims[-3]):
                for j in range(net.dims[-2]):
                    col = cmap_2(j)
                    ax1.plot(i, WYH[j, i], ".", color=col, label=f"to {i}")
                    ax1.plot(i, WIH[j, i], "x", color=col, label=f"from {i}")
            ax1.set_title("Feedforward weights 0")

    ax4.plot(*zip(*net.test_acc))
    ax5.plot(*zip(*net.test_loss))

    ax0.set_title("train loss")
    ax1.set_title("Feedforward error")
    ax2.set_title("Feedback weights")
    ax3.set_title("Feedforward weights")
    ax4.set_title("Test Accuracy")
    ax5.set_title("Test Loss")

    ax0.set_ylim(bottom=0)
    ax4.set_ylim(0, 1)
    ax5.set_ylim(bottom=0)

    plt.savefig(out_file)
    plt.close()
    print(f"done.\n")
