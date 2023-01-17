import numpy as np
import sys
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")

from params import *  # nopep8
from utils import *  # nopep8



dims = [10, 8, 5]


tau_x = neuron_params["tau_x"]
U_x = np.asmatrix(np.zeros(dims[0]))

hx_teacher = np.asmatrix(np.random.random((dims[1], dims[0])) * 2 - 1)
yh_teacher = np.asmatrix(np.random.random((dims[2], dims[1])) * 2 - 1) / neuron_params["gamma"]

y_teacher = []
x_record = []




U_y = np.asmatrix(np.zeros(dims[2]))
y_foo = []


for j in range(6):

    stim = np.random.random(dims[0]) * 2 - 1

    amp_y = phi(yh_teacher * phi(hx_teacher * np.reshape(stim, (-1,1)))).T
    for i in range(1000):

        delta_u_y = -U_y + amp_y
        U_y = U_y + (delta_t/tau_x) * delta_u_y
        y_foo.append(U_y)

        delta_u_x = -U_x + stim
        U_x = U_x + (delta_t/tau_x) * delta_u_x
        y_teacher.append(phi(yh_teacher * phi(hx_teacher * U_x.T)).T)
        x_record.append(U_x)
# x_nest = nest.Create(neuron_params["model"], dims[0], neuron_params["input"])


y_teacher = np.array(y_teacher).squeeze(1)
x_record = np.array(x_record).squeeze(1)
y_foo = np.array(y_foo).squeeze(1)

fig, axes = plt.subplots(dims[-1])

for i in range(dims[-1]):
    axes[i].plot(y_teacher[:, i], label="low-passed input")
    axes[i].plot(y_foo[:, i], label="low-passed output")

axes[0].legend()
plt.show()
