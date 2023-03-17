import nest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import re
from scipy.ndimage import uniform_filter1d
# import torch
from networks.network import Network
import json
from copy import deepcopy


def setup_directories(root="/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/runs", type=""):
    # TODO: remove personal path!
    root = os.path.join(root, f"{type}_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}")

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


# def setup_torch(use_cuda=True):
#     # We don't make use of gradients, so we can save some compute time here.
#     torch.set_grad_enabled(False)

#     device_name = "cpu"
#     if use_cuda:
#         if not torch.cuda.is_available():
#             print("Cuda is not available on this system, computing on CPU")
#         else:
#             device_name = "cuda"
#             torch.set_default_tensor_type('torch.cuda.FloatTensor')

#     device = torch.device(device_name)
#     return device


def rolling_avg(input, size):
    return uniform_filter1d(input, size, mode="nearest")


def store_synaptic_weights(network: Network, dirname, filename="weights.json"):
    if len(network.dims) != 3:
        raise ValueError("I'm too lazy to generalize this!")

    weights = network.get_weight_dict()

    for layer in weights:
        for k, v in layer.items():
            if type(layer[k]) == np.ndarray:
                layer[k] = v.tolist()

    with open(os.path.join(dirname, filename), "w") as f:
        json.dump(weights, f, indent=4)


def setup_models(spiking, nrn, sim, syn, record_weights=False):
    if not spiking:
        nrn["weight_scale"] = 1
        nrn["pyr"]["basal"]["g_L"] = 1
        nrn["pyr"]["apical_lat"]["g_L"] = 1
        nrn["intn"]["basal"]["g_L"] = 1
        nrn["intn"]["apical_lat"]["g_L"] = 1
        nrn["input"]["basal"]["g_L"] = 1
        nrn["input"]["apical_lat"]["g_L"] = 1

    if nrn["latent_equilibrium"]:
        nrn["pyr"]["latent_equilibrium"] = True
        nrn["intn"]["latent_equilibrium"] = True
        nrn["input"]["latent_equilibrium"] = True

    neuron_model = 'pp_cond_exp_mc_pyr' if spiking else 'rate_neuron_pyr'
    nrn["model"] = neuron_model
    syn_model = 'pyr_synapse' if spiking else 'pyr_synapse_rate'
    syn["synapse_model"] = syn_model
    static_syn_model = 'static_synapse' if spiking else 'rate_connection_delayed'

    wr = None
    if record_weights:
        wr = nest.Create("weight_recorder", params={'record_to': "ascii", "precision": 12})
        # wr = nest.Create("weight_recorder")
        nest.CopyModel(syn_model, 'record_syn', {"weight_recorder": wr})
        syn_model = 'record_syn'
        nest.CopyModel(static_syn_model, 'static_record_syn', {"weight_recorder": wr})
        static_syn_model = 'static_record_syn'

    syn_static = {
        "synapse_model": static_syn_model,
        "delay": sim["delta_t"]
    }

    syn_plastic = {
        "synapse_model": syn_model,
        'tau_Delta': syn["tau_Delta"],
        'Wmin': syn["Wmin"] / (nrn["weight_scale"] if spiking else 1),  # minimum weight
        'Wmax': syn["Wmax"] / (nrn["weight_scale"] if spiking else 1),  # maximum weight
        'delay': sim["delta_t"]
    }

    connections = []
    pyr_comps = nest.GetDefaults(neuron_model)["receptor_types"]
    basal_dendrite = pyr_comps['basal']
    apical_dendrite = pyr_comps['apical_lat']
    for layer in range(len(sim["dims"])-2):
        connections_l = {}

        for type in ["up", "pi", "ip"]:
            eta = syn["eta"][type][layer]

            if eta != 0:
                connections_l[type] = deepcopy(syn_plastic)
                connections_l[type]["eta"] = eta
            else:
                connections_l[type] = deepcopy(syn_static)

        connections_l["down"] = deepcopy(syn_static)

        connections_l["up"]['receptor_type'] = basal_dendrite
        connections_l["ip"]['receptor_type'] = basal_dendrite
        connections_l["pi"]['receptor_type'] = apical_dendrite
        connections_l["down"]['receptor_type'] = apical_dendrite
        # connections_l["down"]['delay'] = 2*sim["delta_t"]
        connections.append(connections_l)

    connection_out = {}
    if syn["eta"]["up"][-1] > 0:
        connection_out["up"] = deepcopy(syn_plastic)
        connection_out["up"]["eta"] = syn["eta"]["up"][-1]
    else:
        connection_out["up"] = deepcopy(syn_static)
    connection_out["up"]['receptor_type'] = basal_dendrite
    connections.append(connection_out)

    syn["conns"] = connections
    return wr


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
            dataframes.append(pd.read_csv(file, sep="\s+", comment='#'))

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
