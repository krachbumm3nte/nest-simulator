from abc import abstractmethod

import numpy as np
from src.dataset import BarDataset, MnistDataset
from src.params import Params


class Network:

    def __init__(self, p: Params) -> None:
        self.p = p

        self.mode = p.mode
        if self.mode == "bars":
            self.dims = [9, 30, 3]
            self.train_samples = 3
            self.val_samples = 1
            self.test_samples = 1
            self.bar_dataset = BarDataset()
            self.get_training_data = self.bar_dataset.get_samples
            self.get_val_data = self.bar_dataset.get_samples
            self.get_test_data = self.bar_dataset.get_samples
        elif self.mode == "mnist":
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
        elif self.mode == "teacher":
            self.train_samples = 25
            self.val_samples = 8
            self.test_samples = 8
            self.k_yh = 10         # hidden to output teacher weight scaling factor
            self.k_hx = 1         # input to hidden teacher weight scaling factor
            self.dims_teacher = self.dims
            self.dims_teacher[1:-1] = self.dims_teacher[1:-1]  # TODO: reduce teacher network size
            self.whx_trgt = self.gen_weights(self.dims_teacher[0], self.dims_teacher[1], -1, 1)
            self.wyh_trgt = self.gen_weights(self.dims_teacher[1], self.dims_teacher[2], -1, 1)
            self.y = np.random.random(self.dims_teacher[-1])
            self.k_yh = self.p.k_yh
            self.k_hx = self.p.k_hx
        elif self.mode == "test":
            self.dims = p.dims

        self.p.dims = self.dims
        self.sim_time = self.p.sim_time
        self.dt = self.p.delta_t
        self.sigma_noise = self.p.sigma
        self.record_interval = self.p.record_interval

        self.gamma = self.p.gamma
        self.beta = self.p.beta
        self.theta = self.p.theta
        self.tau_x = self.p.tau_x
        self.le = self.p.latent_equilibrium

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
        x_batch, y_batch = self.get_test_data(self.test_samples)

        acc, loss = self.test_batch(x_batch, y_batch)
        # acc_2, loss_2 = self.test_batch_old(x_batch, y_batch)


        # print(f"acc n/o: {acc:.3f}, {acc_2:.3f}, loss: {loss:.3f}, {loss_2:.3f}")
        self.test_acc.append([self.epoch, acc])
        self.test_loss.append([self.epoch, loss])

    def validate_epoch(self):
        x_batch, y_batch = self.get_val_data(self.val_samples)
        acc, loss = self.test_batch(x_batch, y_batch)

        self.val_acc.append([self.epoch, acc])
        self.val_loss.append([self.epoch, loss])

        self.reset()