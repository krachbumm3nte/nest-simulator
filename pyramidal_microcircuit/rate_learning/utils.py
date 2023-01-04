import pandas as pd
import numpy as np
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
    n_out = len(set(conn.get("target")))
    n_in = len(set(conn.get("source")))
    filtered_data = data[(data.targets.isin(set(conn.target)) & data.senders.isin(set(conn.source)))]
    sorted_data = filtered_data.sort_values(by=["senders", "targets"])["weights"].values
    return np.reshape(sorted_data, (-1, n_out, n_in), "F")


def read_data(device_id, path):
    device_re = f"/it(.+)-{device_id}-(.+)dat"
    files = glob.glob(path + "/*")
    frames = []
    for f in files:
        if re.search(device_re, f):
            frames.append(pd.read_csv(f, sep="\s+", comment='#'))

    return pd.concat(frames)
