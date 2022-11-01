import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative


def phi(x):
    return 1 / (1 + np.exp(-x))


timestep = 0.1


def rate_impl(x):
    rate = 1000 * phi(x)
    return 1 - np.exp(-rate * timestep * 1e-3)


def rate_new(x):
    rate = phi(x)
    return 1 - np.exp(-rate * timestep) + 0.01


x = np.linspace(-1.5, 5, 500)


plt.plot(x, phi(x), label="phi")
plt.plot(x, rate_impl(x), label="rate (old)")
plt.plot(x, rate_new(x), label="rate (new)")

plt.legend()
plt.show()
