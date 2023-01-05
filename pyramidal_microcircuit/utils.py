import nest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import re


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


def setup_simulation():
    # environment parameters
    root = f"/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/runs/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"

    imgdir = os.path.join(root, "plots")
    datadir = os.path.join(root, "data")
    for p in [root, imgdir, datadir]:
        os.mkdir(p)

    return imgdir, datadir


def setup_nest(delta_t, threads, record_interval, datadir):
    nest.set_verbosity("M_ERROR")
    nest.resolution = delta_t
    nest.SetKernelStatus({"local_num_threads": threads, "use_wfr": False})
    nest.rng_seed = 15

    nest.SetDefaults("multimeter", {'interval': record_interval})
    nest.SetKernelStatus({"data_path": datadir})


def read_data(device_id, path):
    device_re = f"/it(.+)-{device_id}-(.+)dat"
    files = glob.glob(path + "/*")
    frames = []
    for f in files:
        if re.search(device_re, f):
            frames.append(pd.read_csv(f, sep="\s+", comment='#'))

    return pd.concat(frames)
