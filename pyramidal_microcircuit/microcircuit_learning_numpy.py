import matplotlib.pyplot as plt
import numpy as np
from params import *
from scipy.ndimage import uniform_filter1d as rolling_avg
import pandas as pd
from networks.network_numpy import NumpyNetwork
from sklearn.metrics import mean_squared_error as mse
from time import time
import utils as utils
import os


imgdir, datadir = utils.setup_simulation()
utils.setup_nest(delta_t, sim_params["threads"], sim_params["record_interval"], datadir)
setup_models(False, False)

dims = [30, 20, 10]

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

plot_interval = 200

T = []
w_pi_errors = []
w_ip_errors = []


net = NumpyNetwork(sim_params, neuron_params, syn_params)


print("setup complete, running simulations...")

for run in range(sim_params["n_runs"] + 1):
    inputs = 2 * np.random.rand(dims[0]) - 1
    # input_index = 0
    net.set_input(inputs)

    start = time()
    net.train(sim_params["SIM_TIME"])
    t = time() - start
    T.append(t)

    if run % plot_interval == 0:
        print(f"plotting run {run}")
        start = time()

        time_progressed = run * sim_params["SIM_TIME"]

        fig, axes = plt.subplots(3, 2, constrained_layout=True)

        [[ax0, ax1], [ax2, ax3], [ax4, ax5]] = axes
        # for ax in axes.flatten():
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.rcParams['savefig.dpi'] = 300

        # plot somatic voltages of hidden interneurons and output pyramidal neurons

        intn_error = np.square(net.U_y_record - net.U_i_record)

        for i in range(dims[2]):
            col = cmap_2(i)
            ax0.plot(rolling_avg(net.U_h_record[:, i], size=250), "--", color=col, alpha=0.5)

            ax0.plot(rolling_avg(net.U_y_record[:, i], size=250), color=col)

            # plot interneuron error
            ax1.plot(rolling_avg(intn_error[:, i], size=150), color=col, alpha=0.35, linewidth=0.7)
        mean_error = rolling_avg(np.sum(intn_error, axis=1), size=250)
        ax1.plot(mean_error, color="black")

        intn_error_now = np.mean(mean_error[-20:])
        ax1_2 = ax1.secondary_yaxis("right")
        ax1_2.set_yticks([intn_error_now])

        # plot apical voltage
        for i in range(dims[1]):
            ax2.plot(rolling_avg(net.V_ah_record[:, i], size=150), label=id)

        # plot apical error
        apical_err = rolling_avg(np.mean(net.V_ah_record, axis=1), size=150)
        ax3.plot(apical_err, label="apical error")
        ax3_2 = ax3.secondary_yaxis("right")
        apical_err_now = np.mean(apical_err[-20:])
        ax3_2.set_yticks([apical_err_now])

        # plot weight error
        why = net.conns["hy"]["record"]
        whi = net.conns["hi"]["record"]

        fb_error = np.mean(np.square(whi + why), axis=(1, 2))
        print(f"int_pyr error: {fb_error[-1]}")
        ax3.plot(fb_error, label=f"FB error: {fb_error[-1]:.3f}")

        wyh = net.conns["yh"]["record"]
        wih = net.conns["ih"]["record"]
        ff_error = np.mean(np.square(wih - wyh), axis=(1, 2))
        print(f"pyr_int error: {ff_error[-1]}")
        ax3.plot(ff_error, label=f"FF error: {ff_error[-1]:.3f}")

        # plot weights
        for i in range(dims[2]):
            col = cmap_2(i)
            for j in range(dims[1]):
                ax4.plot(j, -why[-1, j, i], ".", color=col, label=f"to {t}")
                ax4.plot(j, whi[-1, j, i], "x", color=col, label=f"from {t}")

        for i in range(dims[1]):
            for j in range(dims[2]):
                col = cmap_2(j)
                ax5.plot(i, wyh[-1, j, i], ".", color=col, label=f"to {t}")
                ax5.plot(i, wih[-1, j, i], "x", color=col, label=f"from {t}")

        ax0.set_title("intn(--) and pyr(-) somatic voltages")
        ax1.set_title("interneuron - pyramidal error")
        ax2.set_title("apical compartment voltages")
        # ax3.set_title("apical error")
        ax4.set_title("Feedback weights")
        ax5.set_title("Feedforward weights")

        ax1.set_ylim(bottom=0)
        ax3.set_ylim(bottom=0)
        # ax4.set_ylim(-1, 1)
        # ax5.set_ylim(-1, 1)

        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, prop={'size': 5})

        # ax2.legend(loc='upper right', ncol=dims[1], prop={'size': 5})
        plt.savefig(os.path.join(imgdir, f"{run}.png"))
        plt.close()

        plot_duration = time() - start
        print(f"mean simulation time: {np.mean(T[-50:]):.2f}s. plot time:{plot_duration:.2f}s. \
apical error: {apical_err_now:.2f}.")
        print(
            f"ff error: {ff_error[-1]:.3f}, fb error: {fb_error[-1]:.3f}, interneuron error: {intn_error_now:.2f}\n")
    elif run % 50 == 0:
        print(f"run {run} completed.")
