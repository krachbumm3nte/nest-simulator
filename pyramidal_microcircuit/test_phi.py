import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative


def phi(x):
    return 1e3 * np.log(1 + np.exp(x))


def p_spike(x):
    return -1 * (np.exp(-phi(x) * 0.1 * 1e-3) - 1)


def phi_2(x):
    return np.log(1 + np.exp(x))


def p_spike_2(x):
    return -1 * (np.exp(-phi_2(x) * 0.1) - 1)


x = np.linspace(-1, 2.5, 100)

# plt.plot(x, phi(x), label="phi(x)")

plt.plot(x, p_spike(x), label="old spike freq")
plt.plot(x, phi_2(x), label="phi")
plt.plot(x, p_spike_2(x), label="new spike freq")

plt.legend()
plt.show()
