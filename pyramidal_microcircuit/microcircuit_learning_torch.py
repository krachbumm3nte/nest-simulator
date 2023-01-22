import matplotlib.pyplot as plt
import numpy as np
from params import *
import pandas as pd
from networks.network_torch import TorchNetwork
from sklearn.metrics import mean_squared_error as mse
from time import time
import utils as utils
import os
import torch

imgdir, datadir = utils.setup_simulation()
utils.setup_torch(False)
setup_models(False, False)


net = TorchNetwork(sim_params, neuron_params, syn_params)

dims = sim_params["dims"]

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

plot_interval = 250

T = []
w_pi_errors = []
w_ip_errors = []




print("setup complete, running simulations...")

for run in range(sim_params["n_runs"] + 1):
    inputs = torch.rand(dims[0])
    net.set_input(inputs)

    start = time()
    net.train(inputs, sim_params["SIM_TIME"])
    t = time() - start
    T.append(t)

    if run % plot_interval == 0:
        print(f"plotting run {run}")
        start = time()
        fig, axes = plt.subplots(3, 2, constrained_layout=True)
        [[ax0, ax1], [ax2, ax3], [ax4, ax5]] = axes
        plt.rcParams['savefig.dpi'] = 300

        intn_error = torch.square(net.U_y_record - net.U_i_record)

        mean_error = utils.rolling_avg(torch.sum(intn_error, axis=1).detach(), size=250)
        abs_voltage = torch.mean(torch.cat([net.U_y_record.detach()[-5:], net.U_i_record.detach()[-5:]]))
        ax0.plot(mean_error, color="black")

        intn_error_now = np.mean(mean_error[-20:])
        ax0_2 = ax1.secondary_yaxis("right")
        ax0_2.set_yticks([intn_error_now])

        # plot apical error
        apical_error = utils.rolling_avg(torch.mean(net.V_ah_record, axis=1).detach(), size=150)
        ax1.plot(apical_error)

        apical_error_now = np.mean(apical_error[-20:])
        ax1_2 = ax1.secondary_yaxis("right")
        ax1_2.set_yticks([apical_error_now])

        # plot weight error
        WHY = net.conns["hy"]["record"]
        WHI = net.conns["hi"]["record"]

        fb_error = np.mean(np.square(WHI + WHY), axis=(1, 2))
        ax2.plot(fb_error, label=f"FB error: {fb_error[-1]:.3f}")

        WYH = net.conns["yh"]["record"]
        WIH = net.conns["ih"]["record"]
        ff_error = np.mean(np.square(WIH - WYH), axis=(1, 2))
        ax3.plot(ff_error, label=f"FF error: {ff_error[-1]:.3f}")

        # plot weights
        for i in range(dims[2]):
            col = cmap_2(i)
            for j in range(dims[1]):
                ax4.plot(j, -WHY[-1, j, i], ".", color=col, label=f"to {t}")
                ax4.plot(j, WHI[-1, j, i], "x", color=col, label=f"from {t}")

        for i in range(dims[1]):
            for j in range(dims[2]):
                col = cmap_2(j)
                ax5.plot(i, WYH[-1, j, i], ".", color=col, label=f"to {t}")
                ax5.plot(i, WIH[-1, j, i], "x", color=col, label=f"from {t}")

        ax0.set_title("interneuron - pyramidal error")
        ax1.set_title("apical error")
        ax2.set_title("Feedback error")
        ax3.set_title("Feedforward error")
        ax4.set_title("Feedback weights")
        ax5.set_title("Feedforward weights")

        ax0.set_ylim(bottom=0)
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        ax3.set_ylim(bottom=0)

        plt.savefig(os.path.join(imgdir, f"{run}.png"))
        plt.close()

        plot_duration = time() - start
        print(f"mean simulation time: {np.mean(T[-50:]):.2f}s. plot time:{plot_duration:.2f}s. \
apical error: {apical_error_now:.2f}.")
        print(f"ff error: {ff_error[-1]:.3f}, fb error: {fb_error[-1]:.3f}, interneuron error: {intn_error_now:.2f}, absolute somatic voltage: {abs_voltage}\n")

    elif run % 50 == 0:
        print(f"run {run} completed.")
