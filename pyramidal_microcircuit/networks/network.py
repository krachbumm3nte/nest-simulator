from copy import deepcopy
import numpy as np
from abc import abstractmethod
from .dataset import MnistDataset, BarDataset

class Network:

    def __init__(self, sim, nrn, syns, mode) -> None:
        self.sim = sim  # simulation parameters
        self.nrn = nrn  # neuron parameters
        self.syn = syns  # synapse parameters

        self.mode = mode
        if mode == "bars":
            self.dims = [9, 30, 3]
            self.train_samples = 24
            self.val_samples = 8
            self.test_samples = 8
            self.bar_dataset = BarDataset()
            self.get_training_data = self.bar_dataset.get_samples
            self.get_val_data = self.bar_dataset.get_samples
            self.get_test_data = self.bar_dataset.get_samples
        elif mode == "mnist":
            self.classes = 2
            self.dims = [784, 100, 25, self.classes]
            self.train_samples = 25
            self.val_samples = 8
            self.test_samples = 8

            print("Preparing MNIST train images...", end=" ")
            self.train_dataset = MnistDataset('train', self.classes)
            print("...Done.")

            print("Preparing MNIST validation images...", end=" ")
            self.val_dataset = MnistDataset('val', self.classes)
            print("...Done.")

            print("Preparing MNIST test images...", end=" ")
            self.test_dataset = MnistDataset('test', self.classes)
            print("...Done.")

            print("Shuffling MNIST train, validation & test images...", end=" ")
            np.random.shuffle(self.train_dataset)
            np.random.shuffle(self.val_dataset)
            np.random.shuffle(self.test_dataset)
            self.get_training_data = self.train_dataset.get_samples
            self.get_val_data = self.val_dataset.get_samples
            self.get_test_data = self.test_dataset.get_samples


            print("...Done.")
        elif mode == "teacher":
            self.train_samples = 25
            self.val_samples = 8
            self.test_samples = 8
            self.dims_teacher = sim["dims_teacher"]
            self.whx_trgt = self.gen_weights(self.dims_teacher[0], self.dims_teacher[1], -1, 1)
            self.wyh_trgt = self.gen_weights(self.dims_teacher[1], self.dims_teacher[2], -1, 1)
            self.y = np.random.random(self.dims_teacher[-1])
            self.k_yh = sim["k_yh"]
            self.k_hx = sim["k_hx"]

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
    def get_weight_dict(self):
        pass

    @abstractmethod
    def train_batch(self, x_batch, y_batch):
        pass

    @abstractmethod
    def test_batch(self, x_batch, y_batch):
        pass

    def phi(self, x):
        return self.gamma * np.log(1 + np.exp(self.beta * (x - self.theta)))

    def phi_constant(self, x):
        return np.log(1.0 + np.exp(x))

    def phi_inverse(self, x):
        return (1 / self.beta) * (self.beta * self.theta + np.log(np.exp(x/self.gamma) - 1))

    def reset(self):
        pass


    def generate_teacher_data(self):
        x = np.random.random(self.dims[0])

        return x, self.get_teacher_output(x)

    def get_teacher_output(self, input_currents):
        assert self.teacher
        return self.k_yh * self.wyh_trgt @ self.phi(self.k_hx * self.whx_trgt @ input_currents)

    def train_epoch(self):
        x_batch, y_batch = self.get_training_data(self.train_samples)
        self.train_batch(x_batch, y_batch)
        self.reset()
        self.epoch += 1

    def test_epoch(self):
        x_batch, y_batch = self.get_test_data(self.train_samples)
        acc, loss = self.test_batch(x_batch, y_batch)
        
        self.test_acc.append([self.epoch, acc])
        self.test_loss.append([self.epoch, loss])
        
        self.reset()

    def validate_epoch(self):
        x_batch, y_batch = self.get_val_data(self.train_samples)
        acc, loss = self.test_batch(x_batch, y_batch)
        
        self.val_acc.append([self.epoch, acc])
        self.val_loss.append([self.epoch, loss])

        self.reset()
