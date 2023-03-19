import glob
import json
import os
import re

import numpy as np
import pandas as pd
from networks.network import Network
from scipy.ndimage import uniform_filter1d

import nest


def setup_directories(type, name="default", root=os.path.join(os.getcwd(), "runs"), ):
    root = os.path.join(root, f"{name}_{type}")  # _{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}")

    imgdir = os.path.join(root, "plots")
    datadir = os.path.join(root, "data")

    if os.path.exists(root):
        return False, False, False

    for dir in [root, imgdir, datadir]:
        os.mkdir(dir)

    return root, imgdir, datadir


def setup_nest(params, datadir=os.getcwd()):
    nest.set_verbosity("M_ERROR")
    nest.resolution = params.delta_t
    nest.SetKernelStatus({"local_num_threads": params.threads})
    nest.SetDefaults("multimeter", {'interval': params.record_interval})
    nest.SetKernelStatus({"data_path": datadir})


def rolling_avg(input, size):
    return uniform_filter1d(input, size, mode="nearest")


def store_synaptic_weights(network: Network, dirname, filename="weights.json"):
    weights = network.get_weight_dict()

    for layer in weights:
        for k, v in layer.items():
            if type(layer[k]) == np.ndarray:
                layer[k] = v.tolist()

    with open(os.path.join(dirname, filename), "w") as f:
        json.dump(weights, f, indent=4)


def read_mm(device_id, path, it_min=None, it_max=None):
    device_pattern = re.compile(fr"/it(?P<iteration>\d+)_(.+)-{device_id}-(.+)dat")
    files = glob.glob(path + "/*")
    dataframes = []
    # if "sionlib" in nest.recording_backends:
    #     for file in files:
    #         reader = nestio.NestReader(file)
    #         for datapoint in reader:
    #             if not datapoint.gid in frames:
    #                 print("foo")
    # else:
    for file in sorted(files):
        if result := re.search(device_pattern, file):
            it = int(result.group('iteration'))
            if (it_min and it < it_min) or (it_max and it >= it_max):
                continue
            dataframes.append(pd.read_csv(file, sep=r"\s+", comment='#'))

    return pd.concat(dataframes)


def read_wr(grouped_df, source, target, sim_time, delta_t):

    source_id = sorted(source.global_id)
    target_id = sorted(target.global_id)

    weight_array = np.zeros((int(sim_time/delta_t), len(target_id), len(source_id)))

    for i, id_source in enumerate(source_id):
        for j, id_target in enumerate(target_id):
            group = grouped_df.get_group((id_source, id_target))
            group = group.drop_duplicates("time_ms")
            group = group.set_index("time_ms")
            group = group.reindex(np.arange(0, sim_time, delta_t))
            group = group.fillna(method="backfill").fillna(method="ffill")
            weight_array[:, j, i] = group.weights.values

    return weight_array


def set_nest_weights(sources, targets, weight_array, scaling_factor):
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            nest.GetConnections(source, target).set({"weight": weight_array[j][i] * scaling_factor})


def dump_state(net, filename):
    with open(filename, "w") as f:
        wtf = {
            "dims": net.dims,
            "nrns": nest.GetNodes().get("model"),
            "in": net.input_neurons[0].get(),
            "pyr": net.layers[0].pyr[0].get(),
            "intn": net.layers[0].intn[0].get(),
            "out": net.layers[-1].pyr[0].get(),
            "up0": net.layers[0].up[0].get(),
            "ip0": net.layers[0].ip[0].get(),
            "pi0": net.layers[0].pi[0].get(),
            "down0": net.layers[0].down[0].get(),
            "up1": net.layers[1].up[0].get(),
            "inputs": [f["I_e"] for f in net.input_neurons.get("soma")],
            "targets": [f["I_e"] for f in net.layers[-1].pyr.get("soma")],
            "t": nest.biological_time,
            "s": net.sim_time
        }

        json.dump(wtf, f, indent=4)
