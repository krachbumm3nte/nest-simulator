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
import matplotlib.pyplot as plt

args = sys.argv[1:]

weight_loc = args[0]
params_loc = args[1]
params = Params(params_loc)
params.weight_scale = 1
params.record_interval = 0.1
utils.setup_nest(params, os.path.curdir)
net = NestNetwork(params)
with open(weight_loc) as f:
    wgts = json.load(f)
net.set_all_weights(wgts)


times = np.round(np.logspace(0.01, 2, 10, endpoint=True), 1)


acc = []

for t in times:
    print(t)
    net.p.test_time = t
    net.p.test_delay = max(0.1, round(0.6*t, 1))
    net.epoch = t
    net.test_epoch()


plt.plot(*zip(*sorted(net.test_acc)))
plt.xscale("log")
plt.show()
