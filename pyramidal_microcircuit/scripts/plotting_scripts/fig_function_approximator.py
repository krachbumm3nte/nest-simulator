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

filter_window = 4
if __name__ == "__main__":

    plot_utils.setup_plt()

    curdir = os.path.dirname(os.path.realpath(__file__))

    result_dir = os.path.join(curdir, "../../results/par_study_function_approximator")
    out_file = os.path.join(curdir, "../../data/fig_function_approximator.png")

    dirnames = os.listdir(result_dir)
    acc = []
    loss = []
    train_loss = []
    fig, [ax0, ax1] = plt.subplots(1, 2, sharex=True)

    all_configs = sorted([name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))])

    n_samples = 2
    for config in all_configs:

        params = Params(os.path.join(result_dir, config, "params.json"))
        n = params.dims[1]
        print(f"testing for n={n}")
        params.weight_scale = 250
        params.eta = {
            "up": [0, 0],
            "down": [0, 0],
            "pi": [0, 0],
            "ip": [0, 0]
        }
        utils.setup_nest(params)

        net = NestNetwork(params)
        if not os.path.isfile(os.path.join(result_dir, config, "weights.json")):
            nest.ResetKernel()
            continue
        with open(os.path.join(result_dir, config, "weights.json")) as f:
            wgts = json.load(f)

        net.set_all_weights(wgts)

        net.test_samples = 50


        with open(os.path.join(result_dir, config, "progress.json")) as f:
            progress = json.load(f)

        net.test_epoch()

        print(net.test_acc, net.test_loss)

        acc.append([n, np.mean([i[1] for i in progress["test_acc"][-n_samples:]])])
        loss.append([n, np.mean([i[1] for i in progress["test_loss"][-n_samples:]])])
        nest.ResetKernel()
    train_loss.append([n, np.mean([i[1] for i in progress["train_loss"][-20:]])])
        
    ax0.plot(*zip(*sorted(acc)))
    ax1.plot(*zip(*sorted(loss)))
    ax1.plot(*zip(*sorted(train_loss)))

    ax0.set_title("Accuracy")
    ax1.set_title("Loss")
    
    ax0.set_ylim(bottom=0)
    ax1.set_ylim(bottom=0)

    # plt.show()

    plt.savefig(out_file)
