import matplotlib.pyplot as plt
import nest
from params_rate_test import *
import numpy as np

# this script shows that under correct parametrization, injected noise creates comparable standard deviation in the
# membrane voltage of neurons. Note that the effective standard deviation of the membrane potentials is different due
# to differences in the implementation, yet they are comfortably in the same order of magnitude, which will suffice
# for these experiments.

nest.SetDefaults("multimeter", {"interval": 0.1})

pyr_in = nest.Create(pyr_model_rate, 1, pyr_params)
mm_in = nest.Create("multimeter", 1, {'record_from': ["V_m.s"]})
nest.Connect(mm_in, pyr_in)
gauss = nest.Create("noise_generator", params={'mean': 0, 'std': sigma})
nest.Connect(gauss, pyr_in, syn_spec={"receptor_type": pyr_comps["soma_curr"]})


delta_u = 0
ux = 0
y = []

T = 10000

for i in range(int(T/delta_t)):
    delta_u = -(g_l + g_d + g_a) * ux
    ux = ux + delta_t * delta_u + noise_factor * np.random.standard_normal()
    y.append(ux)

nest.Simulate(T)

print(f"analytical voltage mean: {np.mean(y):.4f}, std:{np.std(y):.4f}")

voltage_nest = mm_in.get("events")["V_m.s"]

print(f"NEST voltage mean: {np.mean(voltage_nest):.4f}, {np.std(voltage_nest):.4f}")

plt.plot(y, label="analytical")
plt.plot(mm_in.get("events")["times"]/delta_t, voltage_nest, label="NEST")
plt.legend()
plt.show()
