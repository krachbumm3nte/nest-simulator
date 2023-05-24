# -*- coding: utf-8 -*-
#
# test_utils.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
import numpy as np


class TestClass(ABC):
    """Abstract base class for all unit tests
    """

    def __init__(self, p, **kwargs) -> None:
        if "record_weights" in kwargs:
            record_weights = kwargs["record_weights"]
        else:
            record_weights = False
        self.p = p
        self.record_interval = 2
        self.tau_x = p.tau_x
        self.delta_t = p.delta_t
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
        self.tau_delta = p.tau_delta
        self.psi = p.psi if self.spiking else 1

        self.lambda_ah = self.g_a / (self.g_d + self.g_a + self.g_l)
        self.lambda_bh = self.g_d / (self.g_d + self.g_a + self.g_l)
        self.lambda_out = self.g_d / (self.g_d + self.g_l)

        p.setup_nest_configs()

    @abstractmethod
    def run(self):
        """Perform the test simulation
        """
        pass

    @abstractmethod
    def evaluate(self) -> bool:
        """Evaluate if the test was successfull

        Returns:
            true, if test was a success, false otherwise
        """
        pass

    @abstractmethod
    def plot_results(self):
        """Generates a matplotlib.pyplot.figure detailing how the test went.
        """
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

    def disable_plasticity(self):
        for conn_type in ["up", "down", "pi", "ip"]:
            self.p.eta[conn_type] = [0 for conn in self.p.eta[conn_type]]


def read_multimeter(mm, key):
    return np.array((mm.events["times"]/0.1, mm.events[key])).swapaxes(0, 1)


def records_match(record_nest, record_numpy, error_threshold=0.005):
    """Checks if records from a NEST simulation and the targeted numpy simulation match

    Arguments:
        record_nest -- NEST-computed results
        record_numpy -- numpy-computed results

    Keyword Arguments:
        error_threshold -- threshold for MSE that separates success from failure (default: {0.005})

    Raises:
        ValueError: If records don't have matching size

    Returns:
        True if match is close enough, false otherwise.
    """
    size_diff = len(record_nest) - len(record_numpy)

    if abs(size_diff) >= 0.1 * len(record_nest):
        raise ValueError(
            f"Arrays have vastly different shapes ({record_nest.shape}, {record_numpy.shape}), cannot compute MSE!")

    if size_diff > 0:
        record_nest = record_nest[:len(record_numpy)]
    elif size_diff < 0:
        record_numpy = record_numpy[:len(record_nest)]

    return mean_squared_error(record_nest, record_numpy) < error_threshold
