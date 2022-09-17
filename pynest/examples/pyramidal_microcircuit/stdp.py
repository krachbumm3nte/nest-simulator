
import math
import matplotlib.pyplot as plt
import numpy as np


tau = 2


def k(x):

    if x < 0:
        return 0
    return 1/tau * math.e ** (-x/tau)


def s_star(x, T):

    out = np.zeros(T)
    for t in range(1, T):
        y = out[t-1]
        if x[t]:
            y += 1
        out[t] = y / tau
    return out

N = 100

x = np.zeros(N, dtype=bool)

for i in [25, 35, 50, 70, 72]:
    x[i] = True

foo = s_star(x, N)

t_ls = 25
t = 30
s_t = (foo[t_ls] + 1/tau) * math.e ** (-(t - t_ls)/tau)
print(s_t, foo[t])

plt.plot(foo)
plt.show()
