import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
import src.utils as utils
from src.params import Params

from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork


if __name__ == "__main__":

    plot_utils.setup_plt()

    args = sys.argv[1:]

    dirname = args[0]
    target_dir = args[1]

    fig, [ax0, ax1, ax2] = plt.subplots(3, 1)

    final_performance = []
    training_duration = []

    p = Params(os.path.join(dirname, "params.json"))
    p.p_conn = 1
    with open(os.path.join(dirname, "progress.json")) as f:
        progress = json.load(f)
    with open(os.path.join(dirname, "weights.json")) as f:
        weights = json.load(f)
    if p.network_type == "numpy":
        net = NumpyNetwork(p)
    else:
        net = NestNetwork(p)

    net.set_all_weights(weights)

    weight_scale = p.weight_scale
    acc = np.array(progress["test_acc"])


    net.test_acc = progress["test_acc"]
    net.test_loss = progress["test_loss"]
    net.train_loss = progress["train_loss"]
    net.ff_error = progress["ff_error"]
    net.fb_error = progress["fb_error"]
    net.intn_error = progress["intn_error"]
    net.apical_error = progress["apical_error"]
    net.epoch = progress["epochs_completed"]    

    # plot_utils.plot_training_progress(net.epoch, net, target_dir)
    plot_utils.plot_pre_training(net.epoch, net, target_dir)