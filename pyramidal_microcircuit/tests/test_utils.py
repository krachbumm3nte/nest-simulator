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

    def __init__(self, p, **kwargs) -> None:
        if "record_weights" in kwargs:
            record_weights = kwargs["record_weights"]
        else:
            record_weights = False
        self.p = p
        self.record_interval = 2
        self.tau_x = p.tau_x
        self.delta_t = p.delta_t
        self.neuron_model = p.neuron_model
        self.g_l_eff = p.g_l_eff
        self.g_a = p.g_a
        self.g_d = p.g_d
        self.g_l = p.g_l
        self.g_som = p.g_som
        self.gamma = p.gamma
        self.beta = p.beta
        self.theta = p.theta
        self.dims = p.dims
        self.spiking = p.spiking
        self.lambda_ah = p.lambda_ah
        self.lambda_bh = p.lambda_bh
        self.lambda_out = p.lambda_out
        self.tau_delta = p.tau_delta
        self.weight_scale = p.weight_scale if self.spiking else 1

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def evaluate(self) -> bool:
        pass

    @abstractmethod
    def plot_results(self):
        pass

    def phi(self, x, thresh=15):
        if not hasattr(x, "__len__"):
            if x > thresh:
                return x
            if x < -thresh:
                x = 0
            return self.gamma * np.log(1 + np.exp(self.beta * (x - self.theta)))
        res = x.copy()
        ind = np.abs(x) < thresh
        res[x < -thresh] = 0
        res[ind] = self.gamma * np.log(1 + np.exp(self.beta * (x[ind] - self.theta)))
        return res

    def phi_inverse(self, x):
        return (1 / self.beta) * (self.beta * self.theta + np.log(np.exp(x/self.gamma) - 1))

    def disable_plasticity(self):
        for conn_type in ["up", "down", "pi", "ip"]:
            self.p.eta[conn_type] = [0 for conn in self.p.eta[conn_type]]

def read_multimeter(mm, key):
    return np.array((mm.events["times"]/0.1, mm.events[key])).swapaxes(0, 1)


def records_match(record_nest, record_numpy, error_threshold=0.005):
    size_diff = len(record_nest) - len(record_numpy)

    if abs(size_diff) >= 0.1 * len(record_nest):
        raise ValueError(
            f"Arrays have vastly different shapes ({record_nest.shape}, {record_numpy.shape}), cannot compute MSE!")

    if size_diff > 0:
        record_nest = record_nest[:len(record_numpy)]
    elif size_diff < 0:
        record_numpy = record_numpy[:len(record_nest)]

    return mean_squared_error(record_nest, record_numpy) < error_threshold
