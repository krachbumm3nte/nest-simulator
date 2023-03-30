import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import src.utils as utils
from src.params import Params
import nest

params = Params()
params.record_interval = 0.1
params.threads = 1
weight_scales = np.logspace(-1, 1, 200, endpoint=True)


acc = []

utils.setup_nest(params)
params.setup_nest_configs()
pyr = nest.Create(params.neuron_model, 1, params.pyr_params)
intn = nest.Create(params.neuron_model, 1, params.intn_params)
#input_nrn = nest.Create(params.neuron_model, 1, params.input_params)

sr = nest.Create("spike_recorder")
nest.Connect(pyr, sr)
nest.Connect(intn, sr)
#nest.Connect(input_nrn, sr)

pyr_hz = []
intn_hz = []
# input_nrn_hz = []
sim_time = 1000


def get_hz_from_sr(spike_recorder, neuron, sim_time):
    n_spikes = len(np.where(spike_recorder.events["senders"] == neuron.global_id)[0])
    return (1000 * n_spikes) / sim_time


for scale in weight_scales:
    sr.n_events = 0
    pyr.gamma = params.gamma * scale
    intn.gamma = params.gamma * scale
    # input_nrn.gamma = scale

    pyr.set({"soma": {"I_e": 0.5 * params.g_som}})
    intn.set({"soma": {"I_e": 0.5 * params.g_som}})
    # input_nrn.set({"soma": {"I_e": 0.5 / params.tau_x}})

    nest.Simulate(sim_time)

    pyr_hz.append([scale, get_hz_from_sr(sr, pyr, sim_time)])
    intn_hz.append([scale, get_hz_from_sr(sr, intn, sim_time)])
    # input_nrn_hz.append([scale, get_hz_from_sr(sr, input_nrn, sim_time)])


min_freq = sorted(pyr_hz)[0]
max_freq = sorted(pyr_hz)[-1]



plt.axes().add_patch(patches.Rectangle([0.1, 200], 10, 600, facecolor="red", alpha = 0.3))


plt.xscale("log")
plt.plot(*zip(*sorted(pyr_hz)), label="Pyramidal neuron")
plt.plot(*zip(*sorted(intn_hz)), label="Interneuron")
# plt.plot(*zip(*sorted(input_nrn_hz)), label="input")
plt.xlabel("weight scale")
plt.ylabel("mean firing rate [Hz]")
plt.legend()
plt.show()
