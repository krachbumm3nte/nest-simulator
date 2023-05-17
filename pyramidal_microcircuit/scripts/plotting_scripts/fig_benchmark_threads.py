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

width = 0.3

labels = {
    "8": {"label": "8",
          "X": 0,
          "color": "blue"},
    "16": {"label": "16",
           "X": 1,
           "color": "blue"},
    "24": {"label": "24",
           "X": 2,
           "color": "blue"},
    "32": {"label": "32",
           "X": 3,
           "color": "blue"},
}

fig, ax = plt.subplots()
all_bars = []
for name, cfg in labels.items():
    bar = ax.bar(cfg["X"], results[name]["t_mean"], color=cfg["color"], label=cfg["label"], width=width*1.5)
    all_bars.append(bar)

ax.set_ylabel(r"$t_{sim} [ms]$")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels([8, 16, 24, 32])

# plt.show()

plt.savefig(out_file)
