import nest
import matplotlib.pyplot as plt
from params import *

# pyr_params["lambda"] = pyr_params["lambda"]
pyr_params["lambda"] = 0.1
pyr_params['basal']['g'] = 1
pyr_params['apical_lat']['g'] = 0
pyr_params['apical_td']['g'] = 0
pyr_params['soma']['g_L'] = 0.6

stim_amp = 10

sim_time = 600

p1 = nest.Create(pyr_model, 1, pyr_params)
mm1 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_td", "V_m.a_lat"]})
p1.set({"soma": {"I_e": stim_amp}})
nest.Connect(mm1, p1)

p2 = nest.Create(pyr_model, 1, pyr_params)
mm2 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_td", "V_m.a_lat"]})
nest.Connect(mm2, p2)

pyr_comps = nest.GetDefaults("pp_cond_exp_mc_pyr")["receptor_types"]
print(pyr_comps)
pyr_id = p2.get("global_id")


nest.Connect(p1, p2, syn_spec={"receptor_type": pyr_comps["basal"], "weight": -1})
nest.Simulate(sim_time)


nest.GetConnections(p1, p2).set({"weight": 0})

p2.set({"soma": {"I_e": stim_amp}})
# p1.target = pyr_id
nest.Simulate(sim_time)
nest.GetConnections(p1, p2).set({"weight": -1})
nest.Simulate(sim_time)


som = mm2.get("events")['V_m.s']
a_lat = mm2.get("events")['V_m.a_lat']
a_td = mm2.get("events")['V_m.a_td']
bas = mm2.get("events")['V_m.b']
plt.plot(som, label="soma p2")
plt.plot(bas, label="basal p2")

som_2 = mm1.get("events")['V_m.s']
# plt.plot(som_2, label="soma p1")
plt.legend()
plt.show()
