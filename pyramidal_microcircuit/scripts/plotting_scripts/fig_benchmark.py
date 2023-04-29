import matplotlib.pyplot as plt
import json
import sys
import numpy as np

with open(sys.argv[1]) as f:
    results = json.load(f)
out_file = sys.argv[2]

n_hidden_list = results.keys()


data = [list(f.values()) for f in results.values()]

X = np.arange(3)
fig, ax = plt.subplots()
ax.bar(X + 0.00, data[0], color='blue', width=0.25)
ax.bar(X + 0.25, data[1], color='orange', width=0.25)
ax.bar(X + 0.50, data[2], color='forestgreen', width=0.25)

ax.set_xticks(X + 0.25)
ax.set_xticklabels(list(results.values())[0].keys())
ax.legend([r"$n_{{hidden}} = {}$".format(n) for n in n_hidden_list])

ax.set_ylabel(r"$T_{{sim}} [s]$")
plt.savefig(out_file)
