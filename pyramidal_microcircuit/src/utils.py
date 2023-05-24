# -*- coding: utf-8 -*-
#
# utils.py
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

import json
import os
import shutil

import numpy as np
import pandas as pd
from src.networks.network import Network
from scipy.ndimage import uniform_filter1d

import nest
import sys


def setup_directories(type, name="default", root=None):
    """Creates directories for storing network training progress

    Arguments:
        type -- Network type - ideally either [numpy, rnest, snest]

    Keyword Arguments:
        name -- Name of the simulation (default: {"default"})
        root -- root directory in which to create folders (default: {None})

    Returns:
        absolute paths for root directory, image subdirectory and data subdirectory
    """
    if root is None:
        root = os.path.join(*[os.path.dirname(os.path.realpath(sys.argv[0])), "..", "results"])

    root = os.path.join(root, f"{name}_{type}")  # _{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}")

    imgdir = os.path.join(root, "plots")
    datadir = os.path.join(root, "data")

    print(f"attemting to create dir {root}")
    if os.path.exists(root):
        reply = input("a simulation of that name already exists, exiting. overwrite? (Y|n)\n")
        if reply in ["y", "Y", "yes", ""]:
            print(f"deleting old contents of {root}")
            shutil.rmtree(root)
        else:
            sys.exit()

    for dir in [root, imgdir, datadir]:
        os.mkdir(dir)

    return root, imgdir, datadir


def setup_nest(params, datadir=os.getcwd()):
    """Sets some critical parameters for the NEST simulator, should be called before running simulations.

    Arguments:
        params -- instance of params.Params

    Keyword Arguments:
        datadir -- directory in which NEST should store multimeter- and spikerecorder data (default: {os.getcwd()})
    """
    nest.set_verbosity("M_ERROR")
    nest.resolution = params.delta_t
    nest.local_num_threads = params.threads
    print(f"configured nest on {nest.local_num_threads} threads")

    if params.record_interval > 0:
        nest.SetDefaults("multimeter", {'interval': params.record_interval})
    nest.SetKernelStatus({"data_path": datadir})


def rolling_avg(input, size):
    return uniform_filter1d(input, size, mode="nearest")


def store_synaptic_weights(network: Network, out_dir):
    """Store full set of network weights to a .json file

    Arguments:
        network -- instance of networks.Network
        out_dir -- full path to target file
    """
    weights = network.get_weight_dict()

    for layer in weights:
        for k, v in layer.items():
            if type(layer[k]) == np.ndarray:
                layer[k] = v.tolist()

    with open(out_dir, "w") as f:
        json.dump(weights, f, indent=4)


def store_progress(net: Network, dirname, epoch, filename="progress.json"):
    """Stores training progress of a network to a .json file

    Arguments:
        net -- instance of networks.Network
        dirname -- directory in which to store file
        epoch -- number of training epochs progressed

    Keyword Arguments:
        filename -- file name I guess (default: {"progress.json"})
    """
    progress = {
        "test_acc": net.test_acc,
        "test_loss": net.test_loss,
        "train_loss": net.train_loss,
        "apical_error": net.apical_error,
        "intn_error": net.intn_error,
        "ff_error": net.ff_error,
        "fb_error": net.fb_error,
        "epochs_completed": epoch
    }
    with open(os.path.join(dirname, filename), "w") as f:
        json.dump(progress, f, indent=4)


def get_mm_data(df, population, key):
    """Reads specified data for a given neuron population from a pandas.DataFrame

    Arguments:
        df -- dataframe of multimeter recordings
        population -- nest.NodeCollection for which data is to be returned
        key -- Key within the dataframe to be returned

    Returns:
        numpy.array with two axes (neuron on first axis, datapoints (sorted by time) on second axis)
    """

    ids = population.global_id
    if hasattr(ids, "__iter__"):  # check if ids is a list/tuple, i.e. contains more than one value
        filtered_df = df[df["senders"].isin(ids)]
        filtered_df = filtered_df.sort_values(by=["times", "senders"])[key]
        return filtered_df.values.reshape(-1, len(ids))
    else:
        filtered_df = df[df["senders"] == ids]
        return filtered_df.sort_values(by="times")[key].values


def read_wr(grouped_df, source, target, t_pres, delta_t=0.1):
    """Reads data from a nest.weight_recorder

    Arguments:
        grouped_df -- pandas.GroupBy object, weight records must be a pandas.DataFrame grouped by (sender,target)
        source -- nest.NodeCollection of source nodes
        target -- nest.NodeCollection of target nodes
        t_pres -- nest.biological_time, i.e. latest possible datapoint

    Keyword Arguments:
        delta_t -- NEST simulator step size (default: {0.1})

    Returns:
        np.array with all synaptic weights between the two populations. Time is on the first axis,
        target is on the second axis, and source on the third axis.
    """

    source_id = source.global_id
    target_id = target.global_id

    if hasattr(source_id, "__iter__"):
        source_id = sorted(source_id)
    else:
        source_id = [source_id]

    if hasattr(target_id, "__iter__"):
        target_id = sorted(target_id)
    else:
        target_id = [target_id]

    weight_array = np.zeros((int(t_pres/delta_t), len(target_id), len(source_id)))

    for i, id_source in enumerate(source_id):
        for j, id_target in enumerate(target_id):
            group = grouped_df.get_group((id_source, id_target))
            group = group.drop_duplicates("times")
            group = group.set_index("times")
            group = group.reindex(np.arange(0, t_pres, delta_t))
            group = group.fillna(method="backfill").fillna(method="ffill")
            weight_array[:, j, i] = group.weights.values

    return weight_array


def generate_weights(dims):
    """Generate a set of weights for arbitrary network dimensions

    Arguments:
        dims -- network dimensions (iterable of lenght equal to the total number of layers)

    Returns:
        dictionary which can be passed to a networks.Network
    """
    weights = []

    for i in range(1, len(dims) - 1):
        weights.append({
            "up": Network.gen_weights(dims[i-1], dims[i]),
            "ip": Network.gen_weights(dims[i], dims[i+1]),
            "pi": Network.gen_weights(dims[i+1], dims[i]),
            "down": Network.gen_weights(dims[i+1], dims[i])
        })
    weights.append({
        "up": Network.gen_weights(dims[-2], dims[-1])
    })
    return weights
