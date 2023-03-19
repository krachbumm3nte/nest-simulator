
import nest
from params import *
import pyramidal_microcircuit.src.utils as utils

utils.setup_nest(sim_params)
wr = utils.setup_models(False, neuron_params, sim_params, syn_params, True)

print(neuron_params["pyr"])
a = nest.Create(neuron_params["model"], 1, neuron_params["pyr"])
b = nest.Create(neuron_params["model"], 1, neuron_params["pyr"])

# conn_1 = nest.Connect(a, b, syn_spec=syn_params["conns"][0]["pi"], return_synapsecollection=True)
conn_1 = nest.Connect(a, b, syn_spec=syn_params["conns"][0]["up"], return_synapsecollection=True)

a.set({"soma": {"I_e": 1}})
nest.Simulate(0.5)
print(wr.get())
