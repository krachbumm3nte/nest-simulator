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
p.dims_teacher = [15, 10, 5]
p.dims = p.dims_teacher
net = NestNetwork(p)
np.set_printoptions(suppress=True)
x, y = net.get_training_data(20000)
print(x[-10:])
print()
print(y[-10:])


print(f"min: {np.min(y)}, max: {np.max(y)}, std: {np.std(y, axis=0)}")