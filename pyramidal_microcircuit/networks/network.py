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
            self.dims_teacher = sim["dims_teacher"]
            self.whx_trgt = self.gen_weights(self.dims_teacher[0], self.dims_teacher[1], -1, 1, True)
            self.wyh_trgt = self.gen_weights(self.dims_teacher[1], self.dims_teacher[2], -1, 1, True)
            self.y = np.random.random(self.dims_teacher[2])
            self.k_yh = sim["k_yh"]
            self.k_hx = sim["k_hx"]
        self.output_loss = []

    def gen_weights(self, n_in, n_out, wmin=None, wmax=None, matrix = False):
        if not wmin:
            wmin = self.syns["wmin_init"]
        if not wmax:
            wmax = self.syns["wmax_init"]
        weights = np.random.uniform(wmin, wmax, (n_out, n_in))
        return np.asmatrix(weights) if matrix else weights

    def calculate_target(self, input_currents):
        assert self.teacher
        self.y = np.squeeze(np.asarray(self.phi(self.k_yh * self.wyh_trgt * self.phi(self.k_hx * self.whx_trgt * np.reshape(input_currents, (-1, 1))))))


    @abstractmethod
    def train(self, input_currents, T):
        pass

    @abstractmethod
    def test(self, input_currents, T):
        pass

    @abstractmethod
    def get_weight_dict(self):
        pass

    def phi(self, x):
        return self.gamma * np.log(1 + np.exp(self.beta * (x - self.theta)))

    def phi_constant(self, x):
        return np.log(1.0 + np.exp(x))


    def phi_inverse(self, x):
        return (1 / self.beta) * (self.beta * self.theta + np.log(np.exp(x/self.gamma) - 1))
