# -*- coding: utf-8 -*-
#
# count_nodes_and_syns.py
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

from src.params import Params
from src.networks.network_nest import NestNetwork
import nest

conf_file = "experiment_configs/mnist_full.json"

p = Params(conf_file)
p.mode = "selfpred"
p.dims = [196, 100, 50, 10]
net = NestNetwork(p)

print(f"Number of neurons: {len(nest.GetNodes())}")
print(f"Number of Connections: {len(nest.GetConnections())}")
print(f"Number of plastic synapses: {len(nest.GetConnections(synapse_model=p.syn_model))}")
