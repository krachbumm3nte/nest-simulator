import time
import matplotlib.pyplot as plt
import numpy as np
from params_rate import *
from scipy.ndimage import uniform_filter1d as rolling_avg
import pandas as pd
from network_mathematica import MathematicaNetwork
from pympler.tracker import SummaryTracker
from sklearn.metrics import mean_squared_error as mse


dims = [30, 20, 10]

cmap_1 = plt.cm.get_cmap('hsv', dims[1]+1)
cmap_2 = plt.cm.get_cmap('hsv', dims[2]+1)

T = []
w_pi_errors = []
w_ip_errors = []


net = MathematicaNetwork(dims)


# tracker = SummaryTracker()

np.seterr('raise')
print("setup complete, running simulations...")

for run in range(n_runs):
    inputs = 2 * np.random.rand(dims[0]) - 1
    # input_index = 0
    net.set_input(inputs)

    start = time.time()
    net.train(SIM_TIME)
    t = time.time() - start
    T.append(t)

    if run % 25 == 0:
        plot_start = time.time()

        time_progressed = run * SIM_TIME

        fig, axes = plt.subplots(3, 2, constrained_layout=True)

        [[ax0, ax1], [ax2, ax3], [ax4, ax5]] = axes

        plt.rcParams['savefig.dpi'] = 300

        # plot somatic voltages of hidden interneurons and output pyramidal neurons

        intn_error = np.square(net.U_y_record - net.U_i_record)

        for i in range(dims[2]):
            ax0.plot(rolling_avg(net.U_h_record[:, i], size=250), "--", color=cmap_2(i), alpha=0.5)

            ax0.plot(rolling_avg(net.U_y_record[:, i], size=250), color=cmap_2(i))

            # plot interneuron error
            ax1.plot(rolling_avg(intn_error[:, i], size=150), color=cmap_2(i), alpha=0.3)

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

        w_ip_error = np.mean(np.square(whi + why), axis=(1, 2))
        print(f"int_pyr error: {w_ip_error[-1]}")
        ax3.plot(w_ip_error, label=f"FB error: {w_ip_error[-1]:.2f}")

        wyh = net.conns["yh"]["record"]
        wih = net.conns["ih"]["record"]
        w_pi_error = np.mean(np.square(wih - wyh), axis=(1, 2))
        print(f"pyr_int error: {w_pi_error[-1]}")
        ax3.plot(w_pi_error, label=f"FF error: {w_pi_error[-1]:.2f}")

        # plot weights
        for i in range(dims[2]):
            col = cmap_2(i)
            for j in range(dims[1]):
                ax4.plot(j, -why[-1, j, i], ".", color=col, label=f"to {t}")
                ax4.plot(j, whi[-1, j, i], "x", color=col, label=f"from {t}")

        for i in range(dims[1]):
            for j in range(dims[2]):
                col = cmap_1(j)
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
        plt.savefig(f"{run}_weights.png")
        plt.close()
        plot_duration = time.time() - plot_start
        print(
            f"{run}: {np.mean(T[-50:]):.2f}s. plot time:{plot_duration:.2f}s apical error: {apical_err_now:.2f}, \
intn error: {intn_error_now:.2f}")
