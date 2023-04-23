from abc import abstractmethod

import numpy as np
from src.dataset import BarDataset, MnistDataset
from src.params import Params


class Network:

    def __init__(self, p: Params) -> None:
        self.p = p

        self.mode = p.mode
        if self.mode == "bars":
            """
            Network is trained on the 'bars' dataset from Haider et al. (2021). Simple classification task
            in which each output neuron represents horizontal, vertical or diagonal alignment respectively
            of the 3x3 input square.
            """
            self.dims = p.dims
            if p.dims[0] != 9 or p.dims[-1] != 3:
                raise ValueError(
                    f"For training on the bar Dataset, network must have exactly 9 input and 3 output neurons, dims are: {p.dims}")
            self.train_samples = 3
            self.val_samples = 1
            self.test_samples = 1
            self.bar_dataset = BarDataset()

            # since the dataset only contains 8 datapoints, all subsets draw from the same function.
            self.get_training_data = self.bar_dataset.get_samples
            self.get_val_data = self.bar_dataset.get_samples
            self.get_test_data = self.bar_dataset.get_samples
        elif self.mode == "mnist":
            """
            Network is trained on the MNIST dataset. Kinda self-explanatory?
            """
            self.n_classes = p.n_classes
            self.dims = p.dims

            self.train_samples = 100
            self.val_samples = 50
            self.test_samples = 10

            print("Preparing MNIST train images...", end=" ")
            self.train_dataset = MnistDataset('train', self.n_classes)
            print("Done.")

            print("Preparing MNIST validation images...", end=" ")
            self.val_dataset = MnistDataset('val', self.n_classes)
            print("Done.")

            print("Preparing MNIST test images...", end=" ")
            self.test_dataset = MnistDataset('test', self.n_classes)
            print("Done.")

            print("Shuffling MNIST train, validation & test images...", end=" ")
            self.train_dataset.shuffle()
            self.val_dataset.shuffle()
            self.test_dataset.shuffle()
            self.get_training_data = self.train_dataset.get_samples
            self.get_val_data = self.val_dataset.get_samples
            self.get_test_data = self.test_dataset.get_samples
            print("Done.")

        elif self.mode == "teacher":
            """
            Network learns to match the input-output relation of a separate, randomly initialized, teacher
            network.
            """
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
        elif self.mode == "self-pred":
            """
            Network learns to reach the self-predicting state.
            """
            self.dims = p.dims
            self.train_samples = 10
            self.test_samples = 5
            self.val_samples = 5

            self.get_training_data = self.generate_selfpred_data
            self.get_val_data = self.generate_selfpred_data
            self.get_test_data = self.generate_selfpred_data
        else:
            self.dims = p.dims

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
        self.apical_error = []
        self.intn_error = []
        self.ff_error = []
        self.fb_error = []

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

    def phi(self, x, thresh=15):

        res = x.copy()
        ind = np.abs(x) < thresh
        res[x < -thresh] = 0
        res[ind] = self.gamma * np.log(1 + np.exp(self.beta * (x[ind] - self.theta)))
        return res

    def phi_constant(self, x):
        return np.log(1.0 + np.exp(x))

    def phi_inverse(self, x):
        return (1 / self.beta) * (self.beta * self.theta + np.log(np.exp(x/self.gamma) - 1))

    def reset(self):
        pass

    def generate_teacher_data(self):
        x = np.random.random(self.dims[0])

        return x, self.get_teacher_output(x)

    def generate_selfpred_data(self, n_samples):
        return np.random.random((n_samples, self.dims[0])), np.zeros((n_samples, self.dims[-1]))

    def get_teacher_output(self, input_currents):
        assert self.teacher
        return self.k_yh * self.wyh_trgt @ self.phi(self.k_hx * self.whx_trgt @ input_currents)

    def train_epoch(self):
        x_batch, y_batch = self.get_training_data(self.train_samples)
        loss = self.train_batch(x_batch, y_batch)
        if loss > 1e4:
            print(f"absurd train loss ({loss})")
        self.train_loss.append((self.epoch, loss))
        self.reset()
        self.epoch += 1

    def test_epoch(self):
        x_batch, y_batch = self.get_test_data(self.test_samples)

        acc, loss = self.test_batch(x_batch, y_batch)
        # acc_2, loss_2 = self.test_batch_old(x_batch, y_batch)

        # print(f"acc n/o: {acc:.3f}, {acc_2:.3f}, loss: {loss:.3f}, {loss_2:.3f}")
        self.reset()
        self.test_acc.append([self.epoch, acc])
        self.test_loss.append([self.epoch, loss])

    def validate_epoch(self):
        x_batch, y_batch = self.get_val_data(self.val_samples)
        acc, loss = self.test_batch(x_batch, y_batch)

        self.val_acc.append([self.epoch, acc])
        self.val_loss.append([self.epoch, loss])

        self.reset()