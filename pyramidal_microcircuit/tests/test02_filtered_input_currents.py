import matplotlib.pyplot as plt
import nest
from params_rate_test import *
import numpy as np

# this script shows that a neuron with attenuated leakage conductance and injected
# current behaves like a low pass filter on injected current. From this, parameters for
# neurons in the input layer can be derived.

pyr_in = nest.Create(pyr_model, 1, pyr_params)
mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
nest.Connect(mm_in, pyr_in)
pyr_in.set({"soma": {"g": tau_input}, "basal": {"g": 0}, "apical_lat": {"g": 0}, "tau_m": tau_input})

delta_u = 0
ux = 0
y = []


sim_times = [50 for i in range(3)]
stim_amps = [2, -2, 0]


for T, amp in zip(sim_times, stim_amps):
    for i in range(int(T/delta_t)):
        delta_u = -ux + amp
        ux = ux + (delta_t/tau_x) * delta_u
        y.append(ux)

    pyr_in.set({"soma": {"I_e": amp*tau_input}})
    nest.Simulate(T)


plt.plot(y, label="exact low pass filtering")
plt.plot(mm_in.get("events")["times"]/delta_t, mm_in.get("events")["V_m.s"], label="pyramidal neuron approximation")
plt.legend()
plt.show()
