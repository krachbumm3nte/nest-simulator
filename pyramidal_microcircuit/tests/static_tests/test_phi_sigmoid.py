import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative

phi_max = 1
gamma = 1
beta = 1
theta = 0

timestep = 0.1


def phi(x):
    return phi_max / (1.0 + gamma * np.exp(beta * (theta - x)))


phi_max_2 = 0.5
gamma_2 = 0.5
beta_2 = 1
theta_2 = 0.5


def phi_new(x):
    return phi_max_2 / (1.0 + gamma_2 * np.exp(beta_2 * (theta_2 - x)))


def rate(x):
    return 1 - np.exp(-x * timestep)


x = np.linspace(-5, 5, 500)

fig, ax = plt.subplots(1, 2)

ax[0].plot(x, phi(x), label="phi (sacramento)")
ax[0].plot(x, phi_new(x), label="phi")

ax[1].plot(x, rate(phi(x)), label="rate sacramento")
ax[1].plot(x, rate(phi_new(x)), label="rate")

ax[0].set_ylim(0)
ax[1].set_ylim(0)

#plt.plot(x, p_spike(x), label="p_spike_1")
#plt.plot(x, p_spike_2(x), label="p_spike_2")

ax[0].legend()
ax[1].legend()
plt.show()
