import os

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
    plt.rcParams['savefig.dpi'] = 500
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['text.usetex'] = True
    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 22}

    # plt.rcParams['font'] = font


def calculate_weight_errors(net, epoch, layer_offset=0):
    # Synaptic weights
    # notice that all weights are scaled up again to ensure that derived metrics are comparible between simulations
    weights = net.get_weight_dict()
    WHI = weights[-(layer_offset + 2)]["pi"]
    WHY = weights[-(layer_offset + 2)]["down"]
    WIH = weights[-(layer_offset + 2)]["ip"]
    WYH = weights[-(layer_offset + 1)]["up"]

    WHI[np.isnan(WHI)] = 0
    WHY[np.isnan(WHY)] = 0
    WIH[np.isnan(WIH)] = 0
    WYH[np.isnan(WYH)] = 0

    fb_error_now = mse(WHY.flatten(), -WHI.flatten())
    net.fb_error.append([epoch, fb_error_now])

    ff_error_now = mse(WYH.flatten(), WIH.flatten())
    net.ff_error.append([epoch, ff_error_now])

    return WHI, WHY, WIH, WYH


def plot_pre_training(epoch, net, out_file):
    print(f"Ep {epoch}: generating plot...", end="")

    cmap_2 = plt.cm.get_cmap('hsv', net.dims[2]+1)

    fig, axes = plt.subplots(3, 2, constrained_layout=True)
    [ax0, ax1, ax2, ax3, ax4, ax5] = axes.flatten()

    WHI, WHY, WIH, WYH = calculate_weight_errors(net, epoch)
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

    intn_error = np.array(net.intn_error)
    print(intn_error)
    ax4.plot(intn_error[:, 0] * net.train_samples, utils.rolling_avg(intn_error[:, 1], 5))
    apical_error = np.array(net.apical_error)
    ax5.plot(apical_error[:, 0] * net.train_samples, utils.rolling_avg(apical_error[:, 1], 5))

    ax0.set_title("Feedback error")
    ax2.set_title("Feedback weights")
    ax3.set_title("Feedforward weights")
    ax4.set_title("Interneuron error")
    ax5.set_title("Apical error")

    ax0.set_ylim(bottom=0)
    ax1.set_ylim(bottom=0)
    ax4.set_ylim(bottom=0)
    ax5.set_ylim(bottom=0)

    plt.savefig(out_file)
    plt.close()
    print(f"done.\n")


def plot_training_progress(epoch, net, out_file):
    print(f"Ep {epoch}: generating plot...", end="")

    cmap_1 = plt.cm.get_cmap('hsv', net.dims[1]+1)
    cmap_2 = plt.cm.get_cmap('hsv', net.dims[2]+1)

    fig, axes = plt.subplots(3, 2, constrained_layout=True)
    [ax0, ax1, ax2, ax3, ax4, ax5] = axes.flatten()

    # Synaptic weights
    # notice that all weights are scaled up again to ensure that derived metrics are comparible between simulations
    WHI, WHY, WIH, WYH = calculate_weight_errors(net, epoch)

    ax0.plot(*zip(*net.train_loss))
    ax1.plot(*zip(*ff_error))

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
        WHI, WHY, WIH, WYH = calculate_weight_errors(net, epoch, 1)

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
    ax5.set_ylim(0, 1)

    plt.savefig(out_file)
    plt.close()
    print(f"done.\n")
