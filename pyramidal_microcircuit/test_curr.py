import nest
import matplotlib.pyplot as plt
from params import *


stim = nest.Create("poisson_generator")
par = nest.Create("parrot_neuron")
nest.Connect(stim, par)
pyr = nest.Create(pyr_model, 1, intn_params)
mm = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_td", "V_m.a_lat"]})
nest.Connect(mm, pyr)

pyr_comps = nest.GetDefaults("pp_cond_exp_mc_pyr")["receptor_types"]
print(pyr_comps)
nest.Connect(par, pyr, syn_spec=syn_laminar_pyr_intn)


stim.rate = 50

nest.Simulate(200)


som = mm.get("events")['V_m.s']
a_lat = mm.get("events")['V_m.a_lat']
a_td = mm.get("events")['V_m.a_td']
bas = mm.get("events")['V_m.b']
plt.plot(som, label="soma")
plt.plot(a_td, label="a_td")
plt.plot(a_lat, label="a_lat")
plt.plot(bas, label="basal")

plt.legend()
plt.show()
