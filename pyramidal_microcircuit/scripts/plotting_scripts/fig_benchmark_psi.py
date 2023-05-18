import matplotlib.pyplot as plt
import json
import src.plot_utils as plot_utils
import os
import numpy as np

plot_utils.setup_plt()

curdir = os.path.dirname(os.path.realpath(__file__))

in_file = os.path.join(curdir, "../../results/benchmarks/benchmark_bars_psi.json")
out_file = os.path.join(curdir, "../../data/fig_benchmark_psi.png")

with open(in_file) as f:
    data = json.load(f)

results = data["results"]

width = 0.4

labels = {
    "psi_1": {"label": r"$\psi = 1$",
              "X": 0,
              "color": "green"},
    "psi_10": {"label":  r"$\psi = 10$",
               "X": 1,
               "color": "green"},
    "psi_100": {"label":  r"$\psi = 100$",
                "X": 2,
                "color": "green"},
    "psi_1000": {"label":  r"$\psi = 1000$",
                 "X": 3,
                 "color": "green"}
}

fig, ax = plt.subplots()
all_bars = []
for name, cfg in labels.items():
    bar = ax.bar(cfg["X"], results[name]["t_mean"], color=cfg["color"], label=cfg["label"], width=width)
    all_bars.append(bar)

ax.set_ylabel(r"$t_{sim}\ [s]$")
ax.set_xlabel(r"$\psi$")
ax.set_xticks(np.arange(4))
ax.set_xticklabels([1, 10, 100, 1000])

# plt.show()

plt.savefig(out_file)
