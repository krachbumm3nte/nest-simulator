# -*- coding: utf-8 -*-
#
# network.py
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

from abc import abstractmethod

import numpy as np
from src.dataset import BarDataset, MnistDataset
from src.params import Params
import json
import os


class Network:
    """Abstract network class for unifying simulations of the dendritic error network.
    """

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
                raise ValueError(f"For training on the bar Dataset, network must have exactly 9 " +
                                 f"input and 3 output neurons, dims are: {p.dims}")
            self.train_samples = 3
            self.val_samples = 1
            self.test_samples = 1
            self.bar_dataset = BarDataset()

            # since the dataset only contains 8 datapoints, all subsets draw from the same function.
            self.get_training_data = self.bar_dataset.get_samples
            self.get_val_data = self.bar_dataset.get_samples
            self.get_test_data = self.bar_dataset.get_samples
        elif self.mode == "bars-flex":
            """
            Network is trained on the 'bars' dataset from Haider et al. (2021). Simple classification task
            in which each output neuron represents horizontal, vertical or diagonal alignment respectively
            of the 3x3 input square.
            """
            self.dims = p.dims
            if p.dims[0] != 9 or p.dims[-1] != 3:
                raise ValueError(f"For training on the bar Dataset, network must have exactly 9 \
input and 3 output neurons, dims are: {p.dims}")
            self.train_samples = 3
            self.val_samples = 1
            self.test_samples = 1
            self.bar_dataset = BarDataset(-0.5, 1)

            self.get_training_data = self.bar_dataset.get_samples
            self.get_val_data = self.bar_dataset.get_samples
            self.get_test_data = self.bar_dataset.get_samples

        elif self.mode == "mnist":
            """
            Network is trained on the MNIST dataset. Kinda self-explanatory?
            """
            self.n_classes = p.n_classes
            self.dims = p.dims

            self.train_samples = 50
            self.val_samples = 10
            self.test_samples = 20

            in_size = p.in_size

            assert (int(in_size**2) == self.dims[0])

            print("Preparing MNIST train images...", end=" ")
            self.train_dataset = MnistDataset('train', self.n_classes, target_size=in_size)
            print("Done.")

            print("Preparing MNIST validation images...", end=" ")
            self.val_dataset = MnistDataset('val', self.n_classes, target_size=in_size)
            print("Done.")

            print("Preparing MNIST test images...", end=" ")
            self.test_dataset = MnistDataset('test', self.n_classes, target_size=in_size)
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
            Network learns to match the input-output function of a separate, randomly initialized teacher
            network.
            """
            self.train_samples = 10
            self.val_samples = 25
            self.test_samples = 50

            # Negative inputs are allowed here, so we need separate stimualtors for SNN
            self.p.add_inhibitory_stims = True

            self.dims = self.p.dims
            self.dims_teacher = self.p.dims_teacher
            if hasattr(p, "teacher_weights"):
                root_dir = os.path.dirname(os.path.realpath(__file__))
                weight_dir = os.path.join(root_dir, "../../experiment_configs/init_weights", p.teacher_weights)
                print(f"Reading teacher weights from: {weight_dir}")
                with open(weight_dir, "r") as f:
                    teacher_weights = json.load(f)
                self.whx_trgt = np.array(teacher_weights[0]["up"])
                self.wyh_trgt = np.array(teacher_weights[1]["up"])
            else:
                self.whx_trgt = self.gen_weights(self.dims_teacher[0], self.dims_teacher[1], -1, 1)
                self.wyh_trgt = self.gen_weights(self.dims_teacher[1], self.dims_teacher[2], -1, 1)
            self.p.wmin_init = -1
            self.p.wmax_init = 1

            self.get_training_data = self.generate_teacher_data
            self.get_val_data = self.generate_teacher_data
            self.get_test_data = self.generate_teacher_data
        elif self.mode == "self-pred":
            """
            Network learns to reach the self-predicting state.
            """
            self.dims = p.dims
            self.train_samples = 10
            self.test_samples = 5
            self.val_samples = 5

            self.p.store_errors = True  # assume that error terms are of interest for self-pred experiments

            self.get_training_data = self.generate_selfpred_data
            self.get_val_data = self.generate_selfpred_data
            self.get_test_data = self.generate_selfpred_data
        else:
            self.dims = p.dims

        self.t_pres = self.p.t_pres
        self.dt = self.p.delta_t
        self.std_noise = self.p.sigma
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
        """Generate a set of weights to initialize a synaptic population

        Arguments:
            n_in -- number of neuron in the source population
            n_out -- number of neurons in the target population

        Keyword Arguments:
            wmin -- minimum initial weight (default: {None})
            wmax -- maximum initial weight (default: {None})

        Returns:
            np.array of dimensions (n_out, n_in)
        """

        if wmin is None:
            wmin = self.p.wmin_init/self.psi
        if wmax is None:
            wmax = self.p.wmax_init/self.psi
        return np.random.uniform(wmin, wmax, (n_out, n_in))

    def phi(self, x, thresh=20):
        """Neuronal transfer function

        Arguments:
            x -- np.array of somatic voltages

        Keyword Arguments:
            thresh -- threshold beyond which the activation scales linearly (default: {15})

        Returns:
            np.array of equal dimensions as the input
        """

        res = x.copy()
        ind = np.abs(x) < thresh
        res[x < -thresh] = 0
        res[ind] = self.gamma * np.log(1 + np.exp(self.beta * (x[ind] - self.theta)))
        return res

    def generate_selfpred_data(self, n_samples):
        """Generate data for training towards the self-predicting state

        Arguments:
            n_samples -- number of samples to generate

        Returns:
            numpy arrays for a batch of stimuli and a batch of targets
        """
        return np.random.random((n_samples, self.dims[0])), np.zeros((n_samples, self.dims[-1]))

    def generate_teacher_data(self, n_samples):
        """Generate data for training to match a separate teacher network

        Arguments:
            n_samples -- number of samples to generate

        Returns:
            numpy arrays for a batch of stimuli and a batch of targets
        """
        x_batch = np.random.random((n_samples, self.dims[0])) * 2 - 1
        y_batch = np.zeros((n_samples, self.dims[-1]))

        for i, x in enumerate(x_batch):
            y_batch[i, :] = self.phi(self.wyh_trgt @ self.phi(self.whx_trgt @ (self.gamma * x)))

        return x_batch, y_batch

    def train_epoch(self):
        """Performs a single training epoch
        """
        x_batch, y_batch = self.get_training_data(self.train_samples)
        loss = self.train_batch(x_batch, y_batch)
        if loss > 1e4:
            print(f"absurd train loss ({loss})")
        self.train_loss.append((self.epoch, loss))
        self.reset()
        self.epoch += 1

    def test_epoch(self):
        """Performs a test epoch
        """
        x_batch, y_batch = self.get_test_data(self.test_samples)
        acc, loss = self.test_batch(x_batch, y_batch)

        self.test_acc.append([self.epoch, acc])
        self.test_loss.append([self.epoch, loss])
        self.reset()

    def validate_epoch(self):
        """Performs a validation epoch
        """
        x_batch, y_batch = self.get_val_data(self.val_samples)
        acc, loss = self.test_batch(x_batch, y_batch)

        self.val_acc.append([self.epoch, acc])
        self.val_loss.append([self.epoch, loss])
        self.reset()

    @abstractmethod
    def set_all_weights(self, weights):
        """Set all network weights from a dictionary containing weight matrices

        Arguments:
            weights -- weight dictionary
        """
        pass

    @abstractmethod
    def get_weight_dict(self):
        """Returns a dictionary of synaptic weights
        """
        pass

    @abstractmethod
    def train_batch(self, x_batch, y_batch):
        """Trains the network on a batch of data

        Arguments:
            x_batch -- numpy.array containing inputs
            y_batch -- numpy.array containing target activations
        """
        pass

    @abstractmethod
    def test_batch(self, x_batch, y_batch):
        """Tests the network on a batch of data

        Arguments:
            x_batch -- numpy.array containing inputs
            y_batch -- numpy.array containing target activations
        """
        pass

    @abstractmethod
    def reset(self):
        pass
