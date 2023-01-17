import numpy as np
import sys
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")

from params import *  # nopep8
from utils import *  # nopep8


setup_models(True, False)
dims = [10, 8, 5]


tau_x = neuron_params["tau_x"]
U_x = np.asmatrix(np.zeros(dims[0]))
U_y = np.asmatrix(np.zeros(dims[2]))
y = np.asmatrix(np.random.random(dims[2]))


hx_teacher = np.asmatrix(np.random.random((dims[1], dims[0])) * 2 - 1)
yh_teacher = np.asmatrix(np.random.random((dims[2], dims[1])) * 2 - 1) / neuron_params["gamma"]

y_record = []
x_record = []


y_nest = nest.Create(neuron_params["model"], dims[-1], neuron_params["pyr"])
mm = nest.Create("multimeter", dims[-1], {'record_from': ["V_m.a_lat", "V_m.s"]})
nest.Connect(mm, y_nest, "one_to_one")


for j in range(6):

    stim = np.random.random(dims[0]) * 2 - 1

    amp_y = phi(yh_teacher * phi(hx_teacher * np.reshape(stim, (-1, 1)))).T
    amp_y = phi_inverse(np.squeeze(np.asarray(amp_y)))
    for n in range(dims[-1]):
        y_nest[n].set({"soma": {"I_e": 0.8 * amp_y[n]}})
    nest.Simulate(100)
    for i in range(1000):

        delta_u_x = -U_x + stim
        delta_u_y = -(g_l + g_d + g_a) * U_y + neuron_params["g_si"] * phi_inverse(y)

        y = phi(yh_teacher * phi(hx_teacher * U_x.T)).T

        U_x = U_x + (delta_t/tau_x) * delta_u_x
        U_y = U_y + delta_t * delta_u_y

        y_record.append(U_y)
        x_record.append(U_x)


y_record = np.array(y_record).squeeze(1)
x_record = np.array(x_record).squeeze(1)

fig, axes = plt.subplots(dims[-1])

for i in range(dims[-1]):
    axes[i].plot(y_record[:, i], label="numpy simulated")
    axes[i].plot(mm[i].get("events")["times"]/delta_t, mm[i].get("events")["V_m.s"], label="Nest simulated")

axes[0].legend()
plt.show()
