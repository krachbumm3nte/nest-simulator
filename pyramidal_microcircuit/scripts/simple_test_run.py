import sys

import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.params import Params

import nest

args = sys.argv[1:]


config = "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/experiment_configs/snn_experiment.json"

params = Params(config)
utils.setup_nest(params, "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/")
print(params.network_type)
params.dims = [5,4,3]
params.mode = "selfpred"
params.spiking = True
net = NestNetwork(params)
nest.Simulate(4)

# net.test_epoch()
