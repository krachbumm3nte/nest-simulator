import nest
import matplotlib.pyplot as plt
from pyramidal_microcircuit.params import *

pyr = nest.Create(pyr_model, 1, pyr_params)
stim = nest.Create("poisson_generator")
par = nest.Create(pyr_model, 1, pyr_params)
sr = nest.Create('spike_recorder')
sr2 = nest.Create('spike_recorder')
mm = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_td", "V_m.a_lat"]})
mm2 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_td", "V_m.a_lat"]})
nest.Connect(mm, pyr)
nest.Connect(mm2, par)

pyr_comps = nest.GetDefaults("pp_cond_exp_mc_pyr")["receptor_types"]

par.target = pyr.get("global_id")

nest.Connect(stim, par, syn_spec={'receptor_type': pyr_comps['soma_exc']})
# nest.Connect(par, pyr, syn_spec={'receptor_type': pyr_comps['soma_curr']})
nest.Connect(par, sr)
nest.Connect(pyr, sr2)

stim.rate = 200

nest.Simulate(300)



som = mm.get("events")['V_m.s']
spar = mm2.get("events")['V_m.s']
#a_lat = mm.get("events")['V_m.a_lat']
#a_td = mm.get("events")['V_m.a_td']
#bas = mm.get("events")['V_m.b']
plt.plot(som, label="soma", color="blue")
plt.plot(spar, label="soma", color="green")
#plt.plot(a_td, label="a_td")
#plt.plot(a_lat, label="a_lat")
#plt.plot(bas, label="basal")
#plt.vlines(sr.get("events")['times'], ymin = 0, ymax = 1, colors= ["red"])
#plt.vlines(sr2.get("events")['times'], ymin = 0, ymax = 1, colors= ["black"])
plt.legend()
plt.show()
