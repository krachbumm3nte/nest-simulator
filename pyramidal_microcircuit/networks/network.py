from copy import deepcopy
import numpy as np
from abc import abstractmethod


class Network:

    def __init__(self, sim, nrn, syns) -> None:
        self.sim = deepcopy(sim)  # simulation parameters
        self.nrn = deepcopy(nrn)  # neuron parameters
        self.syns = deepcopy(syns)  # synapse parameters

        self.dims = sim["dims"]
        self.iteration = 0
        self.sigma_noise = sim["sigma"]

        self.gamma = nrn["gamma"]
        self.beta = nrn["beta"]
        self.theta = nrn["theta"]

        self.teacher = sim["teacher"]
        if self.teacher:
            # TODO: this is wrong!
            self.hx_teacher = self.gen_weights(self.dims[0], self.dims[1], True)
            self.yh_teacher = self.gen_weights(self.dims[1], self.dims[2], True) / self.nrn["gamma"]
            self.y = np.random.random(self.dims[2])
            self.output_loss = []

    def gen_weights(self, lr, next_lr, matrix = False):
        weights = np.random.uniform(self.syns["wmin_init"], self.syns["wmax_init"], (next_lr, lr))
        return np.asmatrix(weights) if matrix else weights

    @abstractmethod
    def train(self, input_currents, T):
        pass

    @abstractmethod
    def test(self, input_currents, T):
        pass

    def phi(self, x):
        return self.gamma * np.log(1 + np.exp(self.beta * (x - self.theta)))

    def phi_constant(self, x):
        return np.log(1.0 + np.exp(x))


    def phi_inverse(self, x):
        return (1 / self.beta) * (self.beta * self.theta + np.log(np.exp(x/self.gamma) - 1))
