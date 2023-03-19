import os

import matplotlib.pyplot as plt
import numpy as np
import utils as utils
from sklearn.metrics import mean_squared_error as mse

ff_error = []
fb_error = []


def plot_progress(epoch, net, imgdir):
    print(f"Ep {epoch}: generating plot...", end="")

    cmap_1 = plt.cm.get_cmap('hsv', net.dims[1]+1)
    cmap_2 = plt.cm.get_cmap('hsv', net.dims[2]+1)

    plt.rcParams['savefig.dpi'] = 300

    fig, axes = plt.subplots(4, 2, constrained_layout=True)
    [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7] = axes.flatten()

    intn_error = np.square(net.U_y_record - net.U_i_record)

    mean_error = utils.rolling_avg(np.mean(intn_error, axis=1), size=200)
    ax0.plot(mean_error, color="black")

    intn_error_now = np.mean(mean_error[-20:])
    ax0_2 = ax0.secondary_yaxis("right")
    ax0_2.set_yticks([intn_error_now])

    apical_error = np.linalg.norm(net.V_ah_record, axis=1)
    ax1.plot(utils.rolling_avg(apical_error, size=150))

    apical_error_now = np.mean(apical_error[-20:])
    ax1_2 = ax1.secondary_yaxis("right")
    ax1_2.set_yticks([apical_error_now])

    # Synaptic weights
    # notice that all weights are scaled up again to ensure that derived metrics are comparible between simulations
    weights = net.get_weight_dict()
    WHI = weights[-2]["pi"]
    WHY = weights[-2]["down"]
    WIH = weights[-2]["ip"]
    WYH = weights[-1]["up"]

    fb_error_now = mse(WHY.flatten(), -WHI.flatten())
    fb_error.append([epoch, fb_error_now])

    ff_error_now = mse(WYH.flatten(), WIH.flatten())
    ff_error.append([epoch, ff_error_now])

    # ax2.plot(*zip(*fb_error), label=f"FB error: {fb_error_now:.3f}")
    ax2.plot(*zip(*net.train_loss))
    ax3.plot(*zip(*ff_error), label=f"FF error: {ff_error_now:.3f}")

    # plot synaptic weights
    for i in range(net.dims[2]):
        col = cmap_2(i)
        for j in range(net.dims[1]):
            ax4.plot(j, -WHY[j, i], ".", color=col, label=f"to {i}")
            ax4.plot(j, WHI[j, i], "x", color=col, label=f"from {i}")

    for i in range(net.dims[1]):
        for j in range(net.dims[2]):
            col = cmap_2(j)
            ax5.plot(i, WYH[j, i], ".", color=col, label=f"to {i}")
            ax5.plot(i, WIH[j, i], "x", color=col, label=f"from {i}")

    ax6.plot(*zip(*net.test_acc))
    ax7.plot(*zip(*net.test_loss))

    ax0.set_title("interneuron - pyramidal error")
    ax1.set_title("apical error")
    ax2.set_title("Feedback error")
    ax3.set_title("Feedforward error")
    ax4.set_title("Feedback weights")
    ax5.set_title("Feedforward weights")
    ax6.set_title("Test Accuracy")
    ax7.set_title("Test Loss")

    ax0.set_ylim(bottom=0)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)
    ax6.set_ylim(0, 1)
    ax7.set_ylim(0, 1)

    plt.savefig(os.path.join(imgdir, f"{epoch}.png"))
    plt.close()
    print(f"done.\n")
