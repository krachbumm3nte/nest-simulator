from copy import deepcopy
import numpy as np
from abc import abstractmethod


class Network:

    def __init__(self, sim, nrn, syns) -> None:
        self.sim = sim  # simulation parameters
        self.nrn = nrn  # neuron parameters
        self.syns = syns  # synapse parameters

        self.dims = sim["dims"]
        self.sim_time = sim["SIM_TIME"]
        self.iteration = 0
        self.sigma_noise = sim["sigma"]

        self.gamma = nrn["gamma"]
        self.beta = nrn["beta"]
        self.theta = nrn["theta"]

        self.teacher = sim["teacher"]
        if self.teacher:
            self.dims_teacher = sim["dims_teacher"]
            self.whx_trgt = self.gen_weights(self.dims_teacher[0], self.dims_teacher[1], -1, 1)
            self.wyh_trgt = self.gen_weights(self.dims_teacher[1], self.dims_teacher[2], -1, 1)
            self.y = np.random.random(self.dims_teacher[-1])
            self.k_yh = sim["k_yh"]
            self.k_hx = sim["k_hx"]

        self.train_loss = []
        self.test_loss = []

    def gen_weights(self, n_in, n_out, wmin=None, wmax=None):
        if not wmin:
            wmin = -0.1
        if not wmax:
            wmax = 0.1
        return np.random.uniform(wmin, wmax, (n_out, n_in))

    def calculate_target(self, input_currents):
        assert self.teacher
        return self.k_yh * self.wyh_trgt @ self.phi(self.k_hx * self.whx_trgt @ input_currents)
    
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

    def generate_bar_data(self, config=None, lo=0.1, hi=1):
        if not config:
            config = np.random.randint(0, 8)
        elif not 0 <= config < 8:
            raise ValueError("Input configuration must be between (0,8]")

        if config == 0:
            input_currents, idx = np.array([[lo, lo, lo],
                                            [lo, lo, lo],
                                            [hi, hi, hi]]), 0
        elif config == 1:
            input_currents, idx = np.array([[lo, lo, lo],
                                            [hi, hi, hi],
                                            [lo, lo, lo]]), 0
        elif config == 2:
            input_currents, idx = np.array([[hi, hi, hi],
                                            [lo, lo, lo],
                                            [lo, lo, lo]]), 0
        elif config == 3:
            input_currents, idx = np.array([[hi, lo, lo],
                                            [hi, lo, lo],
                                            [hi, lo, lo]]), 1
        elif config == 4:
            input_currents, idx = np.array([[lo, hi, lo],
                                            [lo, hi, lo],
                                            [lo, hi, lo]]), 1
        elif config == 5:
            input_currents, idx = np.array([[lo, lo, hi],
                                            [lo, lo, hi],
                                            [lo, lo, hi]]), 1
        elif config == 6:
            input_currents, idx = np.array([[hi, lo, lo],
                                            [lo, hi, lo],
                                            [lo, lo, hi]]), 2
        elif config == 7:
            input_currents, idx = np.array([[lo, lo, hi],
                                            [lo, hi, lo],
                                            [hi, lo, lo]]), 2

        target_currents = np.zeros(3)
        target_currents[idx] = 1

        return input_currents.flatten(), target_currents

    def generate_teacher_data(self):
        x = np.random.random(self.dims[0])

        return x, self.calculate_target(x)
