import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative


def phi(x):
    return 1 / (1 + np.exp(-x))


def p_spike(x):
    return -1 * (np.exp(-phi(x) * 0.1) - 1)


phi_max = 0.9
rate_slope = 0.9
beta = 3.
theta = 1

def phi_old(x):
    return phi_max / ( 1.0 + rate_slope * np.exp( beta * ( theta - x ) ) );



def phi_2(x):
    return np.log(1 + np.exp(x))

def p_spike_2(x):
    return -1 * (np.exp(-phi_2(x) * 0.1) - 1)

x = np.linspace(-1.5, 5, 500)

# plt.plot(x, phi(x), label="phi(x)")

def p_sin(x):
    return phi(np.sin(x))



plt.plot(x, phi(x), label="phi")
plt.plot(x, phi_2(x), label="phi (sacramento)")
plt.plot(x, phi_old(x), label="phi old")

#plt.plot(x, p_spike(x), label="p_spike_1")
#plt.plot(x, p_spike_2(x), label="p_spike_2")

plt.legend()
plt.show()
