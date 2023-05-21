import matplotlib.pyplot as plt
import json
import src.plot_utils as plot_utils
import os

plot_utils.setup_plt()

curdir = os.path.dirname(os.path.realpath(__file__))

in_file = os.path.join(curdir, "../../results/benchmarks/benchmark_mnist_threads.json")
out_file = os.path.join(curdir, "../../data/fig_benchmark_threads.png")

with open(in_file) as f:
    data = json.load(f)

results = data["results"]

width = 0.4

color = "orange"

labels = {
    "8": {"label": "8",
          "X": 0,
          "color": color},
    "16": {"label": "16",
           "X": 1,
           "color": color},
    "24": {"label": "24",
           "X": 2,
           "color": color},
    "32": {"label": "32",
           "X": 3,
           "color": color},
}

fig, ax = plt.subplots(figsize=(4,2.25))
all_bars = []
for name, cfg in labels.items():
    bar = ax.bar(cfg["X"], results[name]["t_mean"], color=cfg["color"], label=cfg["label"], width=width)
    all_bars.append(bar)

ax.set_ylabel(r"$t_{sim}\ [s]$")
ax.set_xlabel(r"Threads")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels([8, 16, 24, 32])

# plt.show()

plt.savefig(out_file)
