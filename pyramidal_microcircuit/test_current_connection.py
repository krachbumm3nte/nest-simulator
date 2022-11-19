import nest
import matplotlib.pyplot as plt
from params import *

# pyr_params["lambda"] = pyr_params["lambda"]

pyr_params["lambda"] = 1
pyr_params['soma']['g_L'] = 0.8
pyr_params['soma']['g'] = 0
pyr_params['basal']['g'] = 0
pyr_params['apical_lat']['g'] = 0
pyr_params['apical_td']['g'] = 0

nest.resolution = 0.1
nest.SetKernelStatus({"local_num_threads": 1})

stim = nest.Create("dc_generator")
par = nest.Create(pyr_model, 1, pyr_params)
nest.Connect(stim, par, syn_spec={"receptor_type": pyr_comps["soma_curr"]})

pyr = nest.Create(pyr_model, 1, pyr_params)
mm = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_td", "V_m.a_lat"]})
nest.Connect(mm, pyr)

mm2 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_td", "V_m.a_lat"]})
nest.Connect(mm2, par)

pyr_comps = nest.GetDefaults("pp_cond_exp_mc_pyr")["receptor_types"]
pyr_id = pyr.get("global_id")
par.target = pyr_id

stim.amplitude = 2
nest.Simulate(20)
stim.amplitude = -2
nest.Simulate(20)
stim.amplitude = 0
nest.Simulate(20)

som = mm.get("events")['V_m.s']
a_lat = mm.get("events")['V_m.a_lat']
a_td = mm.get("events")['V_m.a_td']
bas = mm.get("events")['V_m.b']
plt.plot(som, label="soma")
plt.plot(a_td, label="a_td")
plt.plot(a_lat, label="a_lat")
plt.plot(bas, label="basal")

som_2 = mm2.get("events")['V_m.s']
plt.plot(som_2, label="soma parrot")
plt.legend()
plt.show()
