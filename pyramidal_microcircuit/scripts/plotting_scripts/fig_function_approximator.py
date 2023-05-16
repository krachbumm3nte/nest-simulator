import json
import os
import re
import sys
import nest
import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils
from sklearn.metrics import mean_squared_error as mse
from src.networks.network_nest import NestNetwork
from src.params import Params
import pandas as pd
from sklearn.metrics import explained_variance_score, r2_score

n_samples = 100
test_samples = 250


if __name__ == "__main__":

    plot_utils.setup_plt()

    curdir = os.path.dirname(os.path.realpath(__file__))

    result_dir = os.path.join(curdir, "../../results/par_study_function_approximator")
    out_file = os.path.join(curdir, "../../data/fig_function_approximator.png")


    dirnames = os.listdir(result_dir)
    test_error = []
    test_loss = []
    train_loss = []
    r2_scores = []
    fig, [ax0, ax1, ax2] = plt.subplots(1, 3, sharex=True)

    all_configs = sorted([name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))])

    for config in all_configs:


        if "weights.json" not in os.listdir(os.path.join(result_dir, config)):
            continue

        params = Params(os.path.join(result_dir, config, "params.json"))
        n = params.dims[1]
        print(f"testing for n={n}")
        params.psi = 250
        params.noise = False
        params.sigma = 0
        params.eta = {
            "up": [0, 0],
            "down": [0, 0],
            "pi": [0, 0],
            "ip": [0, 0]
        }
        utils.setup_nest(params)

        net = NestNetwork(params)
        # if not os.path.isfile(os.path.join(result_dir, config, "weights.json")):
        #     nest.ResetKernel()
        #     continue
        with open(os.path.join(result_dir, config, "weights.json")) as f:
            wgts = json.load(f)

        net.set_all_weights(wgts)

        with open(os.path.join(result_dir, config, "progress.json")) as f:
            progress = json.load(f)

        x_batch, y_batch = net.get_test_data(test_samples)

        net_test_acc = []
        net_test_loss = []

        y_pred_total = []

        net.disable_plasticity()
        for i, (x, y) in enumerate(zip(x_batch, y_batch)):
            net.set_input(x)
            net.simulate(net.t_pres, True)
            mm_data = pd.DataFrame.from_dict(net.mm.events)
            U_Y = [mm_data[mm_data["senders"] == out_id]["V_m.s"] for out_id in net.layers[-1].pyr.global_id]
            y_pred = np.mean(U_Y, axis=1)

            net_test_loss.append(mse(y, y_pred))
            net_test_acc.append(np.argmax(y) == np.argmax(y_pred))
            y_pred_total.append(y_pred)
            net.reset()

        print(y_batch.shape)

        test_error.append([n, 1-np.mean(net_test_acc)])
        test_loss.append([n, np.mean(net_test_loss)])
        r2_scores.append([n, r2_score(y_true=y_batch, y_pred=y_pred_total)])
        train_loss.append([n, np.mean([i[1] for i in progress["train_loss"][-n_samples:]])])
        nest.ResetKernel()
        print(test_error[-1], test_loss[-1], r2_scores[-1])

    ax0.plot(*zip(*sorted(test_loss)), label="Test")
    ax0.plot(*zip(*sorted(train_loss)), label="Train")
    ax1.plot(*zip(*sorted(test_error)))
    ax2.plot(*zip(*sorted(r2_scores)), label="explained variance")

    ax0.set_title("Loss")
    ax1.set_title("Test error")
    ax2.set_title("Explained variance")
    ax0.legend()
    ax0.set_ylim(bottom=0)
    ax1.set_ylim(0, 1)
    # ax1.set_ylim(bottom=0)

    # plt.show()

    plt.savefig(out_file)
