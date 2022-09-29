import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative


def phi(x):
    return 1 / (1 + 0.5 * math.e ** (5 * (1-x)))


def lph(x):
    return np.log(phi(x))


def deriv(x):
    return derivative(lph, x)


x = np.linspace(-1, 2.5, 100)

plt.plot(x, phi(x), label="phi(x)")
plt.plot(x, lph(x), label="log(phi(x))")
plt.plot(x, deriv(x), label="h(x)")
plt.legend()
plt.show()
