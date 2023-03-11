from copy import deepcopy
import numpy as np
from abc import abstractmethod


class Network:

    def __init__(self, sim, nrn, syns) -> None:
        self.sim = sim  # simulation parameters
        self.nrn = nrn  # neuron parameters
        self.syn = syns  # synapse parameters

        self.dims = sim["dims"]
        self.sim_time = sim["SIM_TIME"]
        self.dt = sim["delta_t"]
        self.teacher = sim["teacher"]
        self.sigma_noise = sim["sigma"]
        self.record_interval = sim["record_interval"]

        self.gamma = nrn["gamma"]
        self.beta = nrn["beta"]
        self.theta = nrn["theta"]
        self.tau_x = nrn["tau_x"]
        self.le = nrn["latent_equilibrium"]

        self.Wmin = syns["Wmin"]
        self.Wmax = syns["Wmax"]

        self.iteration = 0  # number of times simulate() has been called. mostly used for storing recordings
        self.epoch = 0  # number of training epochs passed

        if self.teacher:
            self.dims_teacher = sim["dims_teacher"]
            self.whx_trgt = self.gen_weights(self.dims_teacher[0], self.dims_teacher[1], -1, 1)
            self.wyh_trgt = self.gen_weights(self.dims_teacher[1], self.dims_teacher[2], -1, 1)
            self.y = np.random.random(self.dims_teacher[-1])
            self.k_yh = sim["k_yh"]
            self.k_hx = sim["k_hx"]

        self.layers = []
        self.train_loss = []
        self.test_loss = []
        self.test_acc = []

    def gen_weights(self, n_in, n_out, wmin=None, wmax=None):
        if not wmin:
            wmin = -0.1
        if not wmax:
            wmax = 0.1
        return np.random.uniform(wmin, wmax, (n_out, n_in))

    @abstractmethod
    def set_all_weights(self, weights):
        pass

    @abstractmethod
    def train(self, input_currents, T):
        pass

    @abstractmethod
    def test(self, input_currents, T):
        pass

    @abstractmethod
    def get_weight_dict(self):
        pass

    @abstractmethod
    def train_epoch(self, x_batch, y_batch):
        pass

    @abstractmethod
    def test_teacher(self):
        pass

    @abstractmethod
    def test_bars(self):
        pass

    def phi(self, x):
        return self.gamma * np.log(1 + np.exp(self.beta * (x - self.theta)))

    def phi_constant(self, x):
        return np.log(1.0 + np.exp(x))

    def phi_inverse(self, x):
        return (1 / self.beta) * (self.beta * self.theta + np.log(np.exp(x/self.gamma) - 1))

    def reset(self):
        pass

    def generate_bar_data(self, config=None, lo=0.1, hi=1):
        """generates a pair of input-output pairings for the "bar" dataset (as described in Haider et al. (2021))

        @note Setting low to 0 makes my simulation terribly inefficient. At this stage, I do not know why that is.

        Keyword Arguments:
            config -- type of bar to generate: horizontal (0-2), vertical (3-5), diagonal (6,7). If unspecified, a random configuration is returned (default: {None})
            lo -- fill value for low signal (default: {0.1})
            hi -- fill value for high signal (default: {1})

        Raises:
            ValueError: if a config outside the specified range is given

        Returns:
            input currents (np.array(3,3)), output currents (np.array(3)) 
        """
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

        return x, self.get_teacher_output(x)

    def get_teacher_output(self, input_currents):
        assert self.teacher
        return self.k_yh * self.wyh_trgt @ self.phi(self.k_hx * self.whx_trgt @ input_currents)

    def train_epoch_bars(self, n_samples=3):
        data_indices = list(range(8)) * n_samples
        np.random.shuffle(data_indices)

        x_batch = np.zeros((len(data_indices), self.dims[0]))
        y_batch = np.zeros((len(data_indices), self.dims[-1]))
        for i, datapoint in enumerate(data_indices):
            x, y = self.generate_bar_data(datapoint)
            x_batch[i] = x
            y_batch[i] = y
        self.train_epoch(x_batch, y_batch)
        self.reset()
        self.epoch += 1

    def train_epoch_teacher(self, batchsize):
        x_batch = np.zeros((batchsize, self.dims[0]))
        y_batch = np.zeros((batchsize, self.dims[-1]))
        for i in range(batchsize):
            x, y = self.generate_teacher_data()
            x_batch[i] = x
            y_batch[i] = y
        self.train_epoch(x_batch, y_batch)
        self.reset()
        self.epoch += 1
