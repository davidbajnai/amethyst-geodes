# This code is used to:
# Compare the stable carbon (d13C) and oxygen (d18O) isotope data
# of the geode calcites with literature data

# INPUT:  UG literature data.csv (data from Gilg et al. (2003) and Morteani et al. (2010))
#         UG Table SX.csv (d18O and d13C data from mass spectrometry)

# OUTPUT: UG Figure 12.png

# >>>>>>>>>

# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot parameters
plt.rcParams.update({"font.size": 6})
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["patch.linewidth"] = 0.6
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'


# Read in carbonate data
df_lit = pd.read_csv(os.path.join(sys.path[0], "UG literature data.csv"))
df_cal = pd.read_csv(os.path.join(sys.path[0], "UG Table S5.csv"))

# Start plotting
fig, ax = plt.subplots()

# Plot literature carbonate samples
categories = ["early", "late"]
markers = {"early": "s", "late": "D"}
for category in categories:
    data = df_lit[df_lit["Type"] == category]
    if len(data) > 0:
        x = data["d18O"]
        y = data["d13C"]
        ax.scatter(x, y, marker=markers[category],
                   fc="#BCBBB2", ec=None, zorder=10)

# Plot carbonate samples
colors = dict(zip(categories, ["#B4D5F6", "#1455C0"]))
for cat in categories:
    for dog in categories:
        dat = df_cal[(df_cal["Type"] == cat) & (df_cal["Type"] == dog)]
        if len(dat) > 0:
            x = dat["d18O"]
            y = dat["d13C"]
            edge_colors = []
            for val in dat["tripleO"]:
                if val == "Y":
                    edge_colors.append("k")
                else:
                    edge_colors.append("w")
            ax.scatter(x, y, marker=markers[cat],
                       color=colors[dog], ec=edge_colors, zorder=10)

# Annotate trends
ax.annotate("", xy=(27, -20), xytext=(27, -0),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1))
ax.text(27.5, -12.5,
        "influence of\norganic carbon",
        rotation=-90, va="center", ha="center")

ax.annotate("", xy=(20, -7.5), xytext=(28, -7.5),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1))
ax.text(20, -7,
        "increasing temperature",
        va="center", ha="left")

# Empty markers for legend
ax.scatter([], [], marker="s", fc="w", ec="k", label="early calcite")
ax.scatter([], [], marker="D", fc="w", ec="k", label="late calcite")
ax.legend(loc="lower left")

# Axis parameters
ax.set_xlabel("$\delta^{18}$O (‰, VSMOW)")
ax.set_ylabel("$\delta^{13}$C (‰, VPDB)")
ax.set_xlim(19, 29)
ax.set_ylim(-21, 1)
ax.set_yticks(np.arange(-20, 1, 2))

plt.savefig(os.path.join(sys.path[0], "UG Figure 12.png"))
