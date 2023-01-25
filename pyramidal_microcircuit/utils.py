# %%
import nest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import re
from scipy.ndimage import uniform_filter1d
import torch
from networks.network import Network
import json


def regroup_records(records, group_key):
    records = pd.DataFrame.from_dict(records)
    return regroup_df(records, group_key)


def regroup_df(df, group_key):
    return dict([(n, x.loc[:, x.columns != group_key]) for n, x in df.groupby(group_key)])


def matrix_from_connection(conn):
    conn_data = conn.get(["weight", "source", "target"])
    if type(conn_data["weight"]) is not list:
        conn_data = [conn_data]  # pandas throws a fit if one-dimensional data isn't indexed. this solves the issue.
    df = pd.DataFrame.from_dict(conn_data)
    n_out = len(set(df["target"]))
    n_in = len(set(df["source"]))
    weights = np.reshape(df.sort_values(by=["source", "target"])["weight"].values, (n_out, n_in), "F")
    return np.asmatrix(weights)


def matrix_from_wr(data, conn):
    t = conn.get("target")
    s = conn.get("source")
    t = {t} if type(t) == int else set(t)
    s = {1} if type(s) == int else set(s)
    filtered_data = data[(data.targets.isin(t) & data.senders.isin(s))]
    sorted_data = filtered_data.sort_values(by=["senders", "targets"])["weights"].values
    return np.reshape(sorted_data, (-1, len(s), len(t)), "F")


def matrix_from_spikes(data, conn, t_max, delta_t):
    t_max = round(t_max/delta_t)
    syns = pd.DataFrame.from_dict(conn.get())
    t = conn.get("target")
    s = conn.get("source")
    t = [t] if type(t) == int else sorted(set(t))
    s = [s] if type(s) == int else sorted(set(s))

    m_idx = pd.MultiIndex.from_product([s, t])
    weight_df = pd.DataFrame(np.full((t_max+1, len(m_idx)), np.nan), index=np.arange(t_max+1), columns=m_idx)

    for i in sorted(s):
        for o in sorted(t):
            grp = data[(data.senders == i) & (data.targets == o)]
            weight_df.loc[grp.times.values, (i, o)] = grp.weights.values
            weight_df.loc[t_max, (i, o)] = syns[(syns.source == i) & (syns.target == o)].weight.iloc[0]
    weight_df = weight_df.fillna(method="bfill")
    return weight_df.values


def setup_simulation():
    # TODO: remove personal path!
    root = f"/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/runs/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"

    imgdir = os.path.join(root, "plots")
    datadir = os.path.join(root, "data")
    for p in [root, imgdir, datadir]:
        os.mkdir(p)

    return imgdir, datadir


def setup_nest(sim_params, datadir=os.getcwd()):
    nest.set_verbosity("M_ERROR")
    nest.resolution = sim_params["delta_t"]
    nest.SetKernelStatus({"local_num_threads": sim_params["threads"]})
    nest.SetDefaults("multimeter", {'interval': sim_params["record_interval"]})
    nest.SetKernelStatus({"data_path": datadir})

    # if "sionlib" in nest.recording_backends:
    #     print("sionlib was found and will be used for recording simulations!")
    #     sim_params["recording_backend"] = "sionlib"
    # else:
    sim_params["recording_backend"] = "ascii"


def setup_torch(use_cuda=True):

    # We don't make use of gradients, so we can save some compute time here.
    torch.set_grad_enabled(False)

    device_name = "cpu"
    if use_cuda:
        if not torch.cuda.is_available():
            print("Cuda is not available on this system, computing on CPU")
        else:
            device_name = "cuda"
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

    device = torch.device(device_name)
    return device


def read_data(device_id, path, it_min=None, it_max=None):
    device_pattern = re.compile(fr"/it(?P<iteration>\d+)_(.+)-{device_id}-(.+)dat")
    
    files = glob.glob(path + "/*")

    frames = []
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
            frames.append(pd.read_csv(file, sep="\s+", comment='#'))

    return pd.concat(frames)


def rolling_avg(input, size):
    return uniform_filter1d(input, size, mode="nearest")


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)

def store_synaptic_weights(net: Network, dirname):
    if len(net.sim["dims"]) != 3:
        raise ValueError("I'm too lazy to generalize this!")

    weights = net.get_weight_dict()

    with open(os.path.join(dirname, "weights.json"), "w") as f:
        json.dump(weights, f)

