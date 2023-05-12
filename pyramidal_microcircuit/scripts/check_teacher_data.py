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

p = Params()
p.mode = "teacher"
p.dims_teacher = [30, 20, 10]
p.dims = p.dims_teacher
net = NestNetwork(p)
np.set_printoptions(suppress=True)
x, y = net.get_training_data(2000)
print(x)
print()
print(y)


print(f"min: {np.min(y)}, max: {np.max(y)}, std: {np.mean(np.std(y, axis=0))}")
