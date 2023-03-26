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


config = "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/experiment_configs/snn_experiment.json"

params = Params(config)
utils.setup_nest(params, "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/")

net = NestNetwork(params)

net.test_epoch()