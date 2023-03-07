import nest
import matplotlib.pyplot as plt
import numpy as np
from networks.network_nest import NestNetwork
from networks.network_numpy import NumpyNetwork
from sklearn.metrics import mean_squared_error as mse
from time import time
import utils
import os
import json
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--network",
                    type=str, choices=["numpy", "rnest", "snest"],
                    default="numpy",
                    help="""Type of network to train. Choice between exact mathematical simulation ('numpy') and NEST simulations with rate- or spiking neurons ('rnest', 'snest')""")
parser.add_argument("--le",
                    action="store_true",
                    help="""Use latent equilibrium in activation and plasticity."""
                    )
parser.add_argument("--cont",
                    type=str,
                    help="""continue training from a previous simulation""")
parser.add_argument("--weights",
                    type=str,
                    help="Start simulations from a given set of weights to ensure comparable results.")
args = parser.parse_args()

run_folder = "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/full_runs"




classes = {
    "rate": [],
    "NEST rate": [],
    "NEST spiking": []
}
runs = os.listdir(run_folder)

for r in runs:
    if r.startswith("numpy"):
        classes["rate"].append(r)
    elif r.startswith("rnest"):
        classes["NEST rate"].append(r)
    elif r.startswith("snest"):
        classes["NEST spiking"].append(r)
    else:
        print(f"could not identify folder: {r}")


fig, axes = plt.subplots(2, 2, sharex=True)

print(classes)

for name, data_dirs in classes.items():
    test_acc = []
    test_loss = []
    train_loss = []
    apical_error = []
    U_y_record = []
    U_i_record = []
    ff_error = []
    fb_error = []

    for root_dir in data_dirs:
        root_dir = os.path.join(run_folder, root_dir)
        imgdir = os.path.join(root_dir, "plots")
        datadir = os.path.join(root_dir, "data")

        with open(os.path.join(root_dir, "weights.json")) as f:
            post_training_weights = json.load(f)
        with open(os.path.join(root_dir, "params.json"), "r") as f:
            all_params = json.load(f)
            sim_params = all_params["simulation"]
            neuron_params = all_params["neurons"]
            syn_params = all_params["synapses"]
        with open(os.path.join(root_dir, "progress.json"), "r") as f:
            progress = json.load(f)

        test_acc.append(progress["test_acc"])
        test_loss.append(progress["test_loss"])
        train_loss.append(progress["train_loss"])
        # apical_error.append(np.linalg.norm(np.array(progress["V_ah_record"]), axis=1))
        # U_y_record.append(np.array(progress["U_y_record"]))
        # U_i_record.append(np.array(progress["U_i_record"]))
        ff_error.append(progress["ff_error"])
        fb_error.append(progress["fb_error"])

    test_acc = np.mean(test_acc, axis=0)
    test_loss = np.mean(test_loss, axis=0)
    train_loss = np.mean(train_loss, axis=0)
    train_loss[:,1] = utils.rolling_avg(train_loss[:,1], size=25)
    # apical_error = np.mean(apical_error, axis=0)
    # U_y_record = np.mean(U_y_record, axis=0)
    # U_i_record = np.mean(U_i_record, axis=0)
    ff_error = np.mean(ff_error, axis=0)
    # fb_error = np.mean(fb_error, axis=0)
    print(name)
    axes[0][0].plot(*zip(*test_acc), label=name)
    axes[0][1].plot(*zip(*test_loss), label=name)
    axes[1][0].plot(*zip(*train_loss), label=name)
    axes[1][1].plot(*zip(*ff_error), label=name)

    axes[0][0].set_title("test Accuracy")
    axes[0][1].set_title("test Loss")
    axes[1][0].set_title("train loss")
    axes[1][1].set_title("FF weight error")

    axes[1][0].set_xlabel("training epoch")
    axes[1][1].set_xlabel("training epoch")


    axes[0][0].legend()
    axes[0][1].legend()
    axes[1][0].legend()
    axes[1][1].legend()


plt.show()
