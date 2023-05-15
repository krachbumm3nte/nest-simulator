import nest
import src.utils as utils
from src.networks.network_nest import NestNetwork
from src.networks.network_numpy import NumpyNetwork
from src.params import Params
import numpy as np
import pandas as pd
import os
import json

p = Params()
p.store_errors = True
p.init_self_pred = False
p.setup_nest_configs()
utils.setup_nest(p)

root_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(root_dir, "../results/par_study_t_pres_le/bars_numpy/bars_le_tpres_50_numpy/weights.json")) as f:
    wgts = json.load(f)

net = NestNetwork(p, wgts)
net.mm.set({"start": 0, "stop": 1e10})
net.disable_plasticity()

net.use_mm = False

a, b = net.get_training_data(1)
for x, y_batch in zip(a, b):
    print(x, y_batch)
    net.set_input(x)
    net.set_target(y_batch)
    nest.Simulate(100)
    net.reset()

data = pd.DataFrame.from_dict(net.mm.get("events"))

v_api = utils.get_mm_data(data, net.layers[0].pyr, "V_m.a_lat")
v_dend = utils.get_mm_data(data, net.layers[0].intn, "V_m.s")
v_bas = utils.get_mm_data(data, net.layers[1].pyr, "V_m.s")

print(np.std(v_api, axis=0))
print(np.mean(np.std(v_api, axis=0)))
print(np.mean(np.array(net.intn_error)[:,1]))
print(np.mean(np.std(v_bas, axis=0)))
