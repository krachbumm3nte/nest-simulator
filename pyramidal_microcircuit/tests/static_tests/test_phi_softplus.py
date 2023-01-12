import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative

timestep = 0.1

gamma = 1.5
beta = 1
theta = 0


def phi(x):
    return gamma * np.log(1 + np.exp(beta * (x - theta)))


gamma_2 = 1
beta_2 = 1
theta_2 = 0


def phi_new(x):
    return gamma_2 * np.log(1 + np.exp(beta_2 * (x - theta_2)))


def rate(x):
    return 1 - np.exp(-x * timestep)


x = np.linspace(-1, 10, 500)

fig, ax = plt.subplots(1, 2)

ax[0].plot(x, phi(x), label="phi (sacramento)")
ax[0].plot(x, phi_new(x), label="phi")

ax[1].plot(x, rate(phi(x)), label="rate sacramento")
ax[1].plot(x, rate(phi_new(x)), label="rate")

#ax[0].set_ylim(0, 1)
#ax[1].set_ylim(0, 1)

#plt.plot(x, p_spike(x), label="p_spike_1")
#plt.plot(x, p_spike_2(x), label="p_spike_2")

ax[0].legend()
ax[1].legend()
plt.show()
