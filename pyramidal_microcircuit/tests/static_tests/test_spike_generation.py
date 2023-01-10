import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
import nest
import sys
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")

from params import *  # nopep8
from utils import *  # nopep8


imgdir, datadir = setup_simulation()
sim_params["record_interval"] = 0.1
sim_params["noise"] = False
sim_params["sigma"] = 0
sim_params["noise_factor"] = 0
sim_params["dims"] = [1, 1, 1]
setup_nest(delta_t, sim_params["threads"], sim_params["record_interval"])
wr = setup_models(True, True)


def p_spike(x):
    return -1 * (np.exp(-phi(x) * 0.1) - 1)


amp = 0.5
weight = .03

delta_t = sim_params["delta_t"]
tau_x = neuron_params["tau_x"]


weight_scale = 14.3
g_lk_dnd = 0.007

weight_scale = 44.3
g_lk_dnd = 0.0023

weight_scale = 1
g_lk_dnd = 0.095

U_x = 0
delta_u_x = 0
V_bh = 0
V_bh_nest = 0

U_x_record = []

V_bh_record = []
V_bh_record_nest = []

for amp in [0.5, 1, 0]:
    for i in range(20000):
        delta_u_x = -U_x + amp

        U_x = U_x + (delta_t/tau_x) * delta_u_x

        # rate based solution
        V_bh = weight * U_x

        # spiking solution
        n_spikes = np.random.poisson(delta_t * U_x)
        V_bh_nest += n_spikes * weight / weight_scale - (g_lk_dnd * V_bh_nest)

        U_x_record.append(U_x)
        V_bh_record.append(V_bh)
        V_bh_record_nest.append(V_bh_nest)


fig, axes = plt.subplots(ncols=2)

axes[0].plot(U_x_record)

axes[1].plot(V_bh_record)
axes[1].plot(V_bh_record_nest, alpha=0.2)
axes[1].plot(rolling_avg(V_bh_record_nest, size=2500))

plt.legend()
plt.show()
