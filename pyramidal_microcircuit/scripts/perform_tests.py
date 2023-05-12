import argparse
import json
import os
import sys
import warnings
from datetime import timedelta
from time import time

import numpy as np
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params
from src.plot_utils import plot_training_progress


args = sys.argv[1:]

root = args[0]
datadir = os.path.join(root, "data")
params = Params(os.path.join(root, "params.json"))
params.psi = 100
params.record_interval = 0.1
utils.setup_nest(params, datadir)
net = NestNetwork(params)

for d in os.listdir(datadir):
    print(f"processing {d}")
    if "weights" not in d:
        continue

    idx = int(d.split("_")[1].split(".")[0])

    net.epoch = idx

    with open(os.path.join(datadir, d)) as f:
        wgts = json.load(f)
    net.set_all_weights(wgts)

    net.test_epoch()

    print(f"idx: {idx}, acc: {net.test_acc[-1][1]}, loss: {net.test_loss[-1][1]}")

with open(os.path.join(root, "test_progress.json"), "w") as f:

    result = {
        "test_acc": sorted(net.test_acc),
        "test_loss": sorted(net.test_loss),
    }

    json.dump(result, f, indent=4)
