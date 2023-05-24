# -*- coding: utf-8 -*-
#
# check_teacher_data.py
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

# validates that the task for matching an arbitrary teacher is difficult enough

import numpy as np
from src.params import Params
from src.networks.network_nest import NestNetwork

p = Params()
p.mode = "teacher"
p.dims_teacher = [30, 20, 10]
p.dims = p.dims_teacher
net = NestNetwork(p)
np.set_printoptions(suppress=True)
x, y_batch = net.get_training_data(20000)
print(x[-10:])
print()
print(y_batch[-8:])

print(f"min: {np.min(y_batch)}, max: {np.max(y_batch)}, std: {np.mean(np.std(y_batch, axis=0))}")
