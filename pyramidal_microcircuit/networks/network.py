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
        self.le = nrn["latent_equilibrium"]
        self.sigma_noise = sim["sigma"]
        self.record_interval = sim["record_interval"]
        self.iteration = 0
        self.tau_x = nrn["tau_x"]


        self.gamma = nrn["gamma"]
        self.beta = nrn["beta"]
        self.theta = nrn["theta"]

        self.Wmin = syns["Wmin"]
        self.Wmax = syns["Wmax"]

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
        self.test_acc = []
        self.conn_names = ["hx", "yh", "ih", "hi", "hy"]


    def gen_weights(self, n_in, n_out, wmin=None, wmax=None):
        if not wmin:
            wmin = -0.1
        if not wmax:
            wmax = 0.1
        return np.random.uniform(wmin, wmax, (n_out, n_in))

    @abstractmethod
    def set_weights(self, weights):
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

        target_currents = np.ones(3) * lo
        target_currents[idx] = hi

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


    def train_epoch_teacher(self, batchsize):
        x_batch = np.zeros((batchsize, self.dims[0]))
        y_batch = np.zeros((batchsize, self.dims[-1]))
        for i in range(batchsize):
            x, y = self.generate_teacher_data()
            x_batch[i] = x
            y_batch[i] = y
        self.train_epoch(x_batch, y_batch)
        self.test_teacher()