import matplotlib.pyplot as plt
import nest
from params_rate_test import *
import numpy as np

# this script shows that a neuron with attenuated leakage conductance and injected
# current behaves like a low pass filter on injected current. From this, parameters for
# neurons in the input layer can be derived.

pyr_model = "pp_cond_exp_mc_pyr"

pyr_in = nest.Create(pyr_model, 1, input_params)
mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
nest.Connect(mm_in, pyr_in)

delta_u = 0
ux = 0
y = []

compartments = nest.GetDefaults(pyr_model)["receptor_types"]

print(compartments)
step_generator = nest.Create("step_current_generator")
print(compartments["soma_curr"])
pyr_in_2 = nest.Create(pyr_model, 1, input_params)
mm_in_2 = nest.Create("multimeter", 1, {'record_from': ["V_m.b", "V_m.s", "V_m.a_lat"]})
nest.Connect(mm_in_2, pyr_in_2)
nest.Connect(step_generator, pyr_in_2, syn_spec={"receptor_type": compartments["soma_curr"]})



sim_times = [50 for i in range(3)]
stim_amps = [2, -2, 0]

step_generator.set(amplitude_values = np.array(stim_amps)*tau_input, amplitude_times = np.cumsum(sim_times).astype(float) - 50 + delta_t)
for T, amp in zip(sim_times, stim_amps):
    for i in range(int(T/delta_t)):
        delta_u = -ux + amp
        ux = ux + (delta_t/tau_x) * delta_u
        y.append(ux)

    pyr_in.set({"soma": {"I_e": amp*tau_input}})
    nest.Simulate(T)


plt.plot(y, label="exact low pass filtering")
plt.plot(mm_in.get("events")["times"]/delta_t, mm_in.get("events")["V_m.s"], label="pyramidal neuron with injected current")
plt.plot(mm_in_2.get("events")["times"]/delta_t, mm_in_2.get("events")["V_m.s"], label="pyramidal neuron with dc generator")
plt.legend()
plt.show()
