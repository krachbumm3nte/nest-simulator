import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative


def p_spike(x):
    return -1 * (np.exp(-phi(x) * 0.1) - 1)


phi_max = 0.15
rate_slope = 0.5
beta = 1/3
theta = -55

phi_max = 1.5
rate_slope = 2
beta = 1
theta = 1

timestep = 0.1


def phi_new(x):
    return phi_max / (1.0 + rate_slope * np.exp(beta * (theta - x)))


def phi(x):
    return 1 / (1 + np.exp(-x))


def rate(x):
    rate = phi(x)
    return 1 - np.exp(-rate * timestep)


def rate_2(x):
    rate = phi_new(x)
    return 1 - np.exp(-rate * timestep)


x = np.linspace(-2, 3, 500)

fig, ax = plt.subplots(1, 2)

ax[0].plot(x, phi(x), label="phi (sacramento)")
ax[0].plot(x, phi_new(x), label="phi")

ax[1].plot(x, rate(x), label="rate sacramento")
ax[1].plot(x, rate_2(x), label="rate")

ax[0].set_ylim(0, 1)
ax[1].set_ylim(0, 1)

#plt.plot(x, p_spike(x), label="p_spike_1")
#plt.plot(x, p_spike_2(x), label="p_spike_2")

ax[0].legend()
ax[1].legend()
plt.show()
