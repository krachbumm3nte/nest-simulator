import nest
import matplotlib.pyplot as plt
import sys
from params_rate_test import *

# pyr_params["lambda"] = pyr_params["lambda"]

pyr_params["lambda"] = 0.5
pyr_params['soma']['g_L'] = 0.8
pyr_params['soma']['g'] = 0
pyr_params['basal']['g'] = 1
pyr_params['apical_lat']['g'] = 0

nest.resolution = 0.1
nest.SetKernelStatus({"local_num_threads": 1})

stim = nest.Create("dc_generator")
pyr_in = nest.Create(pyr_model, 1, pyr_params)
nest.Connect(stim, pyr_in, syn_spec={"receptor_type": pyr_comps["soma_curr"]})
mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
nest.Connect(mm_in, pyr_in)
pyr_in.tau_m = 1


pyr_out = nest.Create(pyr_model, 1, pyr_params)
nest.Connect(stim, pyr_out, syn_spec={"receptor_type": pyr_comps["soma_curr"]})
mm_out = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
nest.Connect(mm_out, pyr_out)
pyr_out.tau_m = 30


pyr_in.set({"basal": {"I_e": 2}})
pyr_out.set({"basal": {"I_e": 2}})
nest.Simulate(20)
pyr_in.set({"basal": {"I_e": -2}})
pyr_out.set({"basal": {"I_e": -2}})
nest.Simulate(20)
pyr_in.set({"basal": {"I_e": 0}})
pyr_out.set({"basal": {"I_e": 0}})
nest.Simulate(20)

som = mm_out.get("events")['V_m.s']
a_lat = mm_out.get("events")['V_m.a_lat']
bas = mm_out.get("events")['V_m.b']
plt.plot(som, label="soma_1")
plt.plot(a_lat, label="a_lat")
plt.plot(bas, label="basal")

som_2 = mm_in.get("events")['V_m.s']
plt.plot(som_2, label="soma_0")
plt.legend()
plt.show()
