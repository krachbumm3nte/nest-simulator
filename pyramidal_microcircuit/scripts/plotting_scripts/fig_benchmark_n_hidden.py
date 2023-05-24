# -*- coding: utf-8 -*-
#
# fig_benchmark_n_hidden.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import json
import src.plot_utils as plot_utils
import os

plot_utils.setup_plt()

curdir = os.path.dirname(os.path.realpath(__file__))

in_file = os.path.join(curdir, "../../results/benchmarks/benchmark_n_hidden.json")
out_file = os.path.join(curdir, "../../data/fig_benchmark_n_hidden.png")

with open(in_file) as f:
    data = json.load(f)

results = data["results"]

width = 0.22
gap = 0.02


x1 = 0.85
labels = {
    "rnest_30": {"label": r"$n_{hidden} = 30$",
                 "X": 0-width-gap,
                 "color": "blue"},
    "rnest_100": {"label": r"$n_{hidden} = 100$",
                  "X": 0,
                  "color": "orange"},
    "rnest_250": {"label": r"$n_{hidden} = 250$",
                  "X": 0+width+gap,
                  "color": "green"},
    "snest_30": {"label": r"$n_{hidden} = 30$",
                 "X": 1-width-gap,
                 "color": "blue"},
    "snest_100": {"label": r"$n_{hidden} = 100$",
                  "X": 1,
                  "color": "orange"},
    "snest_250": {"label": r"$n_{hidden} = 250$",
                  "X": 1+width+gap,
                  "color": "green"},
    "numpy_30": {"label": r"$n_{hidden} = 30$",
                 "X": 2-width-gap,
                 "color": "blue"},
    "numpy_100": {"label": r"$n_{hidden} = 100$",
                  "X": 2,
                  "color": "orange"},
    "numpy_250": {"label": r"$n_{hidden} = 250$",
                  "X": 2+width+gap,
                  "color": "green"},
}

fig, ax = plt.subplots(figsize=(8, 4))
all_bars = []
for name, cfg in labels.items():
    bar = ax.bar(cfg["X"], results[name]["t_mean"], color=cfg["color"], label=cfg["label"], width=width)
    all_bars.append(bar)

ax.set_ylabel(r"$t_{sim}\ [s]$")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["NEST rate", "NEST spiking", "NumPy"])

ax.legend(handles=all_bars[:3])

# plt.show()

plt.savefig(out_file)
