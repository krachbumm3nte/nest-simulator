# -*- coding: utf-8 -*-
#
# dataset.py
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


import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MnistDataset(Dataset):
    def __init__(self, which='train', num_classes=10, n_samples=-1, zero_at=0, one_at=1, target_size=28):
        self.cs = []
        self.vals = []
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # ensure valid parameters
        assert 2 <= num_classes <= 10
        assert 100 <= n_samples <= 5000 or n_samples == -1
        self.target_size = target_size
        if target_size == 28:
            self.transform = None
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(target_size, antialias=True)
            ])

        if which == 'train':
            self.data = datasets.MNIST(root='../data/mnist', train=True, download=True, transform=self.transform)
            if n_samples == -1:
                n_samples = 5000
            for i in range(0, 10 * n_samples):
                if self.data.targets[i] > num_classes - 1:
                    continue
                cs_flat = np.zeros(num_classes)
                cs_flat[self.data.targets[i]] = 1
                cs_flat = zero_at + cs_flat * (one_at - zero_at)
                self.cs.append(cs_flat)
                dat_float = self.data.data[i].float()
                dat_float /= 256.
                dat_float = zero_at + dat_float * (one_at - zero_at)
                self.vals.append(dat_float)
        elif which == 'val':
            self.data = datasets.MNIST(root='../data/mnist', train=True, download=True, transform=self.transform)
            if n_samples == -1:
                n_samples = 1000
            for i in range(50 * n_samples, 60 * n_samples):
                if self.data.targets[i] > num_classes - 1:
                    continue
                cs_flat = np.zeros(num_classes)
                cs_flat[self.data.targets[i]] = 1
                cs_flat = zero_at + cs_flat * (one_at - zero_at)
                self.cs.append(cs_flat)
                dat_float = self.data.data[i].float()
                dat_float /= 256.
                dat_float = zero_at + dat_float * (one_at - zero_at)
                self.vals.append(dat_float)
        elif which == 'test':
            self.data = datasets.MNIST(root='../data/mnist', train=False, download=True, transform=self.transform)
            if n_samples == -1:
                n_samples = 1000
            for i in range(10 * n_samples):
                if self.data.targets[i] > num_classes - 1:
                    continue
                cs_flat = np.zeros(num_classes)
                cs_flat[self.data.targets[i]] = 1
                cs_flat = zero_at + cs_flat * (one_at - zero_at)
                self.cs.append(cs_flat)
                dat_float = self.data.data[i].float()
                dat_float /= 256.
                dat_float = zero_at + dat_float * (one_at - zero_at)
                self.vals.append(dat_float)

        self.vals = [t.numpy() for t in self.vals]
        self.vals = np.array(self.vals)
        self.cs = np.array(self.cs)

    def __getitem__(self, key):

        vals = self.vals[key]
        if self.transform:
            vals = self.transform(vals)
        return vals.flatten(), self.cs[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __len__(self):
        return len(self.cs)

    def get_samples(self, n_samples):
        indices = np.random.choice(len(self.cs), n_samples)

        foo = [self.__getitem__(i) for i in indices]
        if self.target_size != 28:
            vals = [k.numpy() for k, v in foo]
        else:
            vals = [k for k, v in foo]
        keys = [v for k, v in foo]
        return vals, keys

    def shuffle(self):
        np.random.shuffle(self.vals)
        np.random.shuffle(self.cs)


class BarDataset(Dataset):

    def __init__(self, lo=0.1, hi=1):
        """
        @note Setting low to 0 makes my simulation terribly inefficient. At this stage, I do not know why that is.

        Keyword Arguments:
            lo -- fill value for low signal (default: {0.1})
            hi -- fill value for high signal (default: {1})

        Raises:
            ValueError: if a config outside the specified range is given

        Returns:
            input currents (np.array(3,3)), output currents (np.array(3))
        """

        self.stimulus = np.array([
            [lo, lo, lo,
             lo, lo, lo,
             hi, hi, hi],

            [lo, lo, lo,
             hi, hi, hi,
             lo, lo, lo],

            [hi, hi, hi,
             lo, lo, lo,
             lo, lo, lo],

            [hi, lo, lo,
             hi, lo, lo,
             hi, lo, lo],

            [lo, hi, lo,
             lo, hi, lo,
             lo, hi, lo],

            [lo, lo, hi,
             lo, lo, hi,
             lo, lo, hi],

            [hi, lo, lo,
             lo, hi, lo,
             lo, lo, hi],

            [lo, lo, hi,
             lo, hi, lo,
             hi, lo, lo],
        ])

        self.labels = [0, 0, 0, 1, 1, 1, 2, 2]
        self.target_currents = []
        for t in self.labels:
            target_pattern = np.zeros(3)
            target_pattern[t] = 1
            self.target_currents.append(target_pattern)
        self.target_currents = np.array(self.target_currents)

    def __getitem__(self, key):
        return self.stimulus[key], self.target_currents[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __len__(self):
        return 8

    def get_samples(self, n_samples):

        indices = np.arange(n_samples * 8) % 8
        np.random.shuffle(indices)

        return self.stimulus[indices], self.target_currents[indices]

    def get_full_dataset(self):
        return self.stimulus, self.target_currents
