import nest
import argparse
import os
import sys
import time
import json
from datetime import timedelta

import numpy as np
import src.utils as utils
import src.plot_utils as plot_utils
from microcircuit_learning import run_simulations
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params
import matplotlib.pyplot as plt

plot_utils.setup_plt()
p = Params()
p.network_type = "snest"
p.spiking = True
p.setup_nest_configs()
utils.setup_nest(p)


def phi(x, thresh=15):
    return p.gamma * np.log(1 + np.exp(p.beta * (x - p.theta)))


x = np.linspace(-20, 20, 2000)

p.gamma = 1
p.beta = 1
p.theta = 0

plt.plot(x, phi(x), label=f"gam {p.gamma}, bet {p.beta}, the {p.theta}")

p.gamma = 0.1
p.beta = 1
p.theta = 3

plt.plot(x, phi(x), label=f"gam {p.gamma}, bet {p.beta}, the {p.theta}")

p.gamma = 1
p.beta = 0.4
p.theta = 1

plt.plot(x, phi(x), label=f"gam {p.gamma}, bet {p.beta}, the {p.theta}")
plt.legend()
plt.show()
