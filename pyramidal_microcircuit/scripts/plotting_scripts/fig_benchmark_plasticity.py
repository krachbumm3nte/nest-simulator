import matplotlib.pyplot as plt
import json
import src.plot_utils as plot_utils
import os

plot_utils.setup_plt()

curdir = os.path.dirname(os.path.realpath(__file__))

in_file = os.path.join(curdir, "../../results/benchmarks/benchmark_plasticity.json")
out_file = os.path.join(curdir, "../../data/fig_benchmark_plasticity.png")

with open(in_file) as f:
    data = json.load(f)

results = data["results"]

width = 0.15


x1 = 0.85
labels = {
    "rnest_plast": {"label": "plastic",
                    "X": 0 + width,
                    "color": "blue",
                    "linestyle": "-"},
    "rnest_static": {"label": "non-plastic",
                     "X": 0 - width,
                     "color": "orange",
                     "linestyle": "--"},
    "snest_plast": {"label": "plastic",
                    "X": x1 + width,
                    "color": "blue",
                    "linestyle": "-"},
    "snest_static": {"label": "non-plastic",
                     "X": x1 - width,
                     "color": "orange",
                     "linestyle": "--"},
}

fig, ax = plt.subplots(figsize=(8, 3.5))
all_bars = []
for name, cfg in labels.items():
    bar = ax.bar(cfg["X"], results[name]["t_mean"], color=cfg["color"], label=cfg["label"], width=width*1.5)
    all_bars.append(bar)

ax.set_ylabel(r"$t_{sim}\ [s]$")
ax.set_xticks([0, x1])
ax.set_xticklabels(["Rate neurons", "Spiking neurons"])

ax.legend(handles=all_bars[1:3])

# plt.show()

plt.savefig(out_file)
