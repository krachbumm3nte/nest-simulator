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
from copy import deepcopy


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


def setup_simulation(root="/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/runs"):
    # TODO: remove personal path!
    root = os.path.join(root, datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))

    imgdir = os.path.join(root, "plots")
    datadir = os.path.join(root, "data")
    for dir in [root, imgdir, datadir]:
        os.mkdir(dir)

    return root, imgdir, datadir


def setup_nest(sim_params, datadir=os.getcwd()):
    nest.set_verbosity("M_ERROR")
    nest.resolution = sim_params["delta_t"]
    nest.SetKernelStatus({"local_num_threads": sim_params["threads"]})
    nest.SetDefaults("multimeter", {'interval': sim_params["record_interval"]})
    nest.SetKernelStatus({"data_path": datadir})


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

    for k,v in weights.items():
        weights[k] = v.tolist()

    with open(os.path.join(dirname, "weights.json"), "w") as f:
        json.dump(weights, f)


def setup_models(spiking, nrn, sim, syn, record_weights=False):
    if not spiking:
        nrn["pyr"]["basal"]["g_L"] = 1
        nrn["pyr"]["apical_lat"]["g_L"] = 1
        nrn["intn"]["basal"]["g_L"] = 1
        nrn["intn"]["apical_lat"]["g_L"] = 1
        nrn["input"]["basal"]["g_L"] = 1
        nrn["input"]["apical_lat"]["g_L"] = 1

    neuron_model = 'pp_cond_exp_mc_pyr' if spiking else 'rate_neuron_pyr'
    nrn["model"] = neuron_model
    syn_model = 'pyr_synapse' if spiking else 'pyr_synapse_rate'
    static_syn_model = 'static_synapse'
    wr = None
    if record_weights:
        wr = nest.Create("weight_recorder", params={'record_to': "ascii", "precision":12})
        nest.CopyModel(syn_model, 'record_syn', {"weight_recorder": wr})
        syn_model = 'record_syn'
        nest.CopyModel(static_syn_model, 'static_record_syn', {"weight_recorder": wr})
        static_syn_model = 'static_record_syn'

    syn["synapse_model"] = syn_model

    syn_static = {
        "synapse_model": static_syn_model,
        "delay": sim["delta_t"]
    }

    for syn_name in ["hx", "yh", "hy", "ih", "hi"]:
        if syn[syn_name]["eta"] > 0:
            syn[syn_name].update({'synapse_model': syn_model})
        else:
            # if learning rate is zero, we can save a lot of compute time by utilizing the static
            # synapse type.
            syn[syn_name] = deepcopy(syn_static)

    pyr_comps = nest.GetDefaults(neuron_model)["receptor_types"]
    basal_dendrite = pyr_comps['basal']
    apical_dendrite = pyr_comps['apical_lat']
    syn["hx"]['receptor_type'] = basal_dendrite
    syn["yh"]['receptor_type'] = basal_dendrite
    syn["ih"]['receptor_type'] = basal_dendrite
    syn["hy"]['receptor_type'] = apical_dendrite
    syn["hi"]['receptor_type'] = apical_dendrite

    return wr

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