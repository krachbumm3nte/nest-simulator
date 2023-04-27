import numpy as np
from src.params import Params
import src.plot_utils as plot_utils
import src.utils as utils
import matplotlib.pyplot as plt
import sys
import json
from src.networks.network_numpy import NumpyNetwork
from src.networks.network_nest import NestNetwork
from copy import deepcopy
import nest

conf_file = "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/experiment_configs/mnist_full.json"

p = Params(conf_file)
net = NestNetwork(p)

print(f"Number of neurons: {len(nest.GetNodes())}")
print(f"Number of Connections: {len(nest.GetConnections())}")
print(f"Number of plastic synapses: {len(nest.GetConnections(synapse_model=p.syn_model))}")
