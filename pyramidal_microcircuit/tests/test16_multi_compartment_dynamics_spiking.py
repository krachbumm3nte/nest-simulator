import nest
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")

from params import *  # nopep8
from utils import *  # nopep8
from networks.network_nest import NestNetwork  # nopep8
from networks.network_numpy import NumpyNetwork  # nopep8
"""
This script shows that the neuron model handles membrane voltage updates exactly as described in the analytical case.
The neuron in the hidden layer (pyr_h) recieves synaptic input to both apical and basal dendrites.
"""

imgdir, datadir = setup_simulation()
sim_params["record_interval"] = 1.5
sim_params["noise"] = False
sim_params["delta_t"] = delta_t
sim_params["teacher"] = False


weight_scale = 250

neuron_params["gamma"] = weight_scale
neuron_params["pyr"]["gamma"] = weight_scale
neuron_params["intn"]["gamma"] = weight_scale
neuron_params["input"]["gamma"] = weight_scale
setup_nest(sim_params, datadir)
wr = setup_models(True, True)

pyr_x = nest.Create(neuron_params["model"], 1, input_params)
mm_x = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_x, pyr_x)

pyr_h = nest.Create(neuron_params["model"], 1, pyr_params)
mm_h = nest.Create("multimeter", 1, {'record_from': ["V_m.s", "V_m.b", "V_m.a_lat"]})
nest.Connect(mm_h, pyr_h)

pyr_y = nest.Create(neuron_params["model"], 1, pyr_params)
mm_y = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_y, pyr_y)


w0 = 1
syn_params["hx"].update({"weight": w0/weight_scale, "eta": 0})
nest.Connect(pyr_x, pyr_h, syn_spec=syn_params["hx"])

w1 = 0.5
syn_params["hy"].update({"weight": w1/weight_scale, "eta": 0})
nest.Connect(pyr_y, pyr_h, syn_spec=syn_params["hy"])


U_x = 0
U_y = 0
U_h = 0
V_bh = 0
V_ah = 0

sim_times = [250 for i in range(3)]
stim_amps = [2, 1, 0.5]
target_amps = [0.1, 0.5, 1]

SIM_TIME = sum(sim_times)


UX = [0]
UH = [0]
UY = [0]
VBH = [0]
VAH = [0]

g_s = neuron_params["g_s"]
tau_x = neuron_params["tau_x"]
for T, stim, target in zip(sim_times, stim_amps, target_amps):
    pyr_x.set({"soma": {"I_e": stim / tau_x}})
    pyr_y.set({"soma": {"I_e": phi_inverse(target) * g_s}})
    nest.Simulate(T)

    for i in range(int(T/delta_t)):

        delta_u_i = -U_x + stim
        delta_u_h = -(g_l + g_d + g_a) * U_h + V_bh * g_d + V_ah * g_a
        delta_u_y = -(g_l + g_d + g_a) * U_y + g_s * phi_inverse(target)

        V_bh = U_x * w0
        V_ah = phi(U_y) * w1
        U_x = U_x + (delta_t/tau_x) * delta_u_i

        U_y = U_y + delta_t * delta_u_y

        U_h = U_h + delta_t * delta_u_h

        UX.append(U_x)
        UH.append(U_h)
        UY.append(U_y)
        VBH.append(V_bh)
        VAH.append(V_ah)


fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, sharex=True)

ax0.plot(mm_x.get("events")["times"]/delta_t, mm_x.get("events")['V_m.s'], label="NEST computed")
ax0.plot(UX, label="analytical")

ax1.plot(mm_h.get("events")['times']/delta_t, mm_h.get("events")['V_m.b'], label="NEST computed")
ax1.plot(VBH, label="analytical")

ax2.plot(mm_h.get("events")["times"]/delta_t, mm_h.get("events")['V_m.s'], label="NEST computed")
ax2.plot(UH, label="analytical")

ax3.plot(mm_h.get("events")['times']/delta_t, mm_h.get("events")['V_m.a_lat'], label="NEST computed")
ax3.plot(VAH, label="analytical")

ax4.plot(mm_y.get("events")["times"]/delta_t, mm_y.get("events")['V_m.s'], label="NEST computed")
ax4.plot(UY, label="analytical")

ax0.set_title("input neuron voltage")
ax1.set_title("hidden neuron basal voltage")
ax2.set_title("hidden neuron somatic voltage")
ax3.set_title("hidden neuron apical voltage")
ax4.set_title("output neuron somatic voltage")

ax0.legend()
plt.show()
