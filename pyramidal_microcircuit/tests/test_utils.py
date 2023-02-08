from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import nest
from sklearn.metrics import mean_squared_error
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")
from networks.network_nest import NestNetwork  # nopep8
from networks.network_numpy import NumpyNetwork  # nopep8
import utils  # nopep8


class TestClass(ABC):

    def __init__(self, nrn, sim, syn, spiking_neurons, **kwargs) -> None:
        if "record_weights" in kwargs:
            record_weights = kwargs["record_weights"]
        else:
            record_weights = False
        self.wr = utils.setup_models(spiking_neurons, nrn, sim, syn, record_weights)
        self.nrn = nrn
        self.sim = sim
        self.syn = syn
        self.tau_x = nrn["tau_x"]
        self.delta_t = sim["delta_t"]
        self.neuron_model = nrn["model"]
        self.g_l_eff = nrn["g_l_eff"]
        self.g_a = nrn["g_a"]
        self.g_d = nrn["g_d"]
        self.g_l = nrn["g_l"]
        self.g_si = nrn["g_si"]
        self.gamma = nrn["gamma"]
        self.beta = nrn["beta"]
        self.theta = nrn["theta"]
        self.dims = sim["dims"]
        self.spiking_neurons = spiking_neurons
        self.lambda_ah = nrn["lambda_ah"]
        self.lambda_bh = nrn["lambda_bh"]
        self.lambda_out = nrn["lambda_out"]
        self.tau_delta = syn["tau_Delta"]
        self.weight_scale = nrn["weight_scale"] if spiking_neurons else 1

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def evaluate(self) -> bool:
        pass

    @abstractmethod
    def plot_results(self):
        pass

    def phi(self, x):
        return self.gamma * np.log(1 + np.exp(self.beta * (x - self.theta)))

    def phi_inverse(self, x):
        return (1 / self.beta) * (self.beta * self.theta + np.log(np.exp(x/self.gamma) - 1))


def read_multimeter(mm, key):
    return np.array((mm.events["times"]/0.1, mm.events[key])).swapaxes(0, 1)


def records_match(record_nest, record_numpy, error_threshold=0.005):
    size_diff = len(record_nest) - len(record_numpy)

    if abs(size_diff) >= 0.1 * len(record_nest):
        raise ValueError("Difference between arrays is too large, cannot compute MSE!")

    if size_diff > 0:
        record_nest = record_nest[:len(record_numpy)]
    elif size_diff < 0:
        record_numpy = record_numpy[:len(record_nest)]

    return mean_squared_error(record_nest, record_numpy) < error_threshold
