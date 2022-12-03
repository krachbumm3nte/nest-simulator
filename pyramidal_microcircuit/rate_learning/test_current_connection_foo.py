import nest
import matplotlib.pyplot as plt
from params_rate import *

# pyr_params["lambda"] = pyr_params["lambda"]
nest.SetKernelStatus({"local_num_threads": 1})

pyr_params["lambda"] = 1
pyr_params['soma']['g_L'] = 0.6


stim = nest.Create("dc_generator")
par = nest.Create("rate_neuron_pyr", 1)
nest.Connect(stim, par, syn_spec={"receptor_type": pyr_comps["soma_curr"]})

mm_stim = nest.Create("multimeter")

pyr = nest.Create("rate_neuron_pyr", 1, pyr_params)
mm = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
nest.Connect(mm, pyr)

mm2 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
nest.Connect(mm2, par)

pyr_comps = nest.GetDefaults("pp_cond_exp_mc_pyr")["receptor_types"]
print(pyr_comps)
nest.Connect(par, pyr, syn_spec=syn_ff_pyr_pyr)

stim.amplitude = 2
nest.Simulate(100)
stim.amplitude = -2
nest.Simulate(100)
stim.amplitude = 0
nest.Simulate(100)

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
