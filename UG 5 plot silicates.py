# This code is used to:
# Plot the geode silicate data in triple oxygen isotope space

# INPUT:  UG Table S3.csv (silicate data)
#         UG fluid model amethyst.csv (best-fit compositions)
#         meteoric_water.csv (modern meteoric water data)
#         UG Table S2.csv (measured geode water data)

# OUTPUT: UG Figure 11A.png

# >>>>>>>>>

# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *

# Plot parameters
plt.rcParams.update({"font.size": 6})
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["patch.linewidth"] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'

# Define functions

def a18_qz(T):

    # Sharp et al. (2016) - Eq. 9
    return np.exp((4.28 * 10**6 / T**2 - 3.5 * 10**3 / T) / 1000)


def theta_qz(T):
    
    # Sharp et al. (2016) - Eq. 10
    return -1.85 / T + 0.5305


def a17_qz(T):
    return a18_qz(T)**theta_qz(T)


def d18Oqz(equilibrium_temperatures, d18Ow):
    return a18_qz(equilibrium_temperatures) * (d18Ow+1000) - 1000


def d17Oqz(equilibrium_temperatures, d18Ow):
    return a17_qz(equilibrium_temperatures) * (d18Ow+1000) - 1000


def plot_quartz_equilibrium(Dp17Ow, d18Ow, Tmin, Tmax, ax, fluid_name="precipitating fluid", color="k"):

    d17Ow = unprime(0.528 * prime(d18Ow) + Dp17Ow/1000)

    ax.scatter(prime(d18Ow), Dp17O(d17Ow, d18Ow),
            marker="X", fc=color, ec="k", zorder=3, label=fluid_name)

    # equilibrium, entire T range
    toInf = np.arange(0, 300, 1) + 273.15
    d18O_mineral = d18Oqz(toInf, d18Ow)
    d17O_mineral = d17Oqz(toInf, d17Ow)
    mineral_equilibrium = np.array(
        [d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), toInf]).T
    ax.plot(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
            ":", c=color, zorder=3, label="equilibrium SiO$_2$")

    # equilibrium, highlight range
    equilibrium_temperatures = np.arange(Tmin, Tmax, 0.5) + 273.15
    colors = np.linspace(0, 1, len(equilibrium_temperatures))
    d18O_mineral = d18Oqz(equilibrium_temperatures, d18Ow)
    d17O_mineral = d17Oqz(equilibrium_temperatures, d17Ow)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(
        d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
    ax.scatter(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
               marker=".", c=colors, cmap='coolwarm', zorder=3)

    # equilibrium, highlight range, marker every 10 °C
    equilibrium_temperatures = np.arange(Tmin, Tmax+1, 10) + 273.15
    d18O_mineral = d18Oqz(equilibrium_temperatures, d18Ow)
    d17O_mineral = d17Oqz(equilibrium_temperatures, d17Ow)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(
        d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
    ax.scatter(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
               s=15, marker="o", fc="white", ec=color, zorder=3)

    # Return equilibrium data as a dataframe
    equilibrium_df = pd.DataFrame(mineral_equilibrium)
    equilibrium_df[2] = equilibrium_df[2]-273.15
    equilibrium_df = equilibrium_df.rename(
        columns={0: 'd18O', 1: 'Dp17O', 2: 'temperature'})
    return equilibrium_df


# Read in data
df_qz = pd.read_csv(os.path.join(sys.path[0], "UG Table S3.csv"))
df_qz["Dp17O"] = Dp17O(df_qz["d17O"], df_qz["d18O"])
df_meteoric_w = pd.read_csv(os.path.join(sys.path[0], "meteoric_water.csv"))
df_geode_w = pd.read_csv(os.path.join(sys.path[0], "UG Table S2.csv"))
df_modeled_fluids = pd.read_csv(os.path.join(sys.path[0], "UG fluid model amethyst.csv"))

# Start plotting
fig, ax = plt.subplots()

# Plot meteoric waters
ax.scatter(prime(df_meteoric_w["d18O"]), Dp17O(df_meteoric_w["d17O"], df_meteoric_w["d18O"]),
           marker="+", fc="#cacaca", zorder=-3,
           label="modern meteoric waters")

# Plot measured geode waters
confidence_ellipse(prime(df_geode_w["d18O"]), df_geode_w["Dp17O"], ax,
                   ec="#005752", zorder=1, lw=1)
ax.scatter(prime(df_geode_w["d18O"].mean()), df_geode_w["Dp17O"].mean(),
           fc="#005752", marker="X", ec="k", zorder=10,
           label="measured geode water")

# Plot modeled silicate fluids
confidence_ellipse(prime(df_modeled_fluids["d18Ow"]), df_modeled_fluids["Dp17Ow"], ax,
                   ec="#F8AB37", lw=1)
dfeq = plot_quartz_equilibrium(df_modeled_fluids["Dp17Ow"].mean(), df_modeled_fluids["d18Ow"].mean(), 10, 60,
                               ax, fluid_name="modeled fluid: amethyst", color="#F8AB37")

# annotate the first and last point in dfeq
ax.annotate(str(round(dfeq.iloc[0, 2])) + " °C", (prime(dfeq.iloc[0, 0]), dfeq.iloc[0, 1]),
            xytext=(prime(dfeq.iloc[0, 0])-5, dfeq.iloc[0, 1]), ha = "right",
            arrowprops=dict(arrowstyle="->", color="#4F4B41"))
ax.annotate(str(round(dfeq.iloc[-1, 2])) + " °C", (prime(dfeq.iloc[-1, 0]), dfeq.iloc[-1, 1]),
            xytext=(prime(dfeq.iloc[-1, 0])-5, dfeq.iloc[-1, 1]), ha = "right",
            arrowprops=dict(arrowstyle="->", color="#4F4B41"))


# Plot silicate samples
cat1 = df_qz["Type"].unique()
markers = dict(zip(cat1, ["s", "^", "h"]))
cat2 = df_qz["Type"].unique()
colors = dict(zip(cat2, ["#FF7A00", "#814997", "#FFF876"]))

for cat in cat1:
    for dog in cat2:
        dat = df_qz[(df_qz["Type"] == cat) & (df_qz["Type"] == dog)]
        if len(dat) > 0:
            x = prime(dat["d18O"])
            y = dat["Dp17O"]
            ax.scatter(x, y, marker=markers[cat],
                    color=colors[dog], label=f"{dog}", edgecolors="k", zorder = 10)

# Plot error bar
ax.errorbar(-20, -120, xerr=0.1, yerr=10, fmt="none",
             color="gray", elinewidth=0.6, zorder=0)
ax.text(-19.5, -120, "mean error of\nsilicate $\Delta\prime^{17}$O", fontsize=6, va="center", ha="left", color="gray")

ax.set_xlabel("$\delta\prime^{18}$O (‰, VSMOW)")
ax.set_ylabel("$\Delta\prime^{17}$O (ppm)")

ax.text(0.02, 0.98, "a", fontsize=14, fontweight="bold",
         va="top", ha="left", transform=ax.transAxes,
         bbox=dict(fc="w", pad=0.1, ec="none", alpha=0.8))

ax.set_ylim(-155, 105)
ax.set_xlim(-25, 45)
ax.legend(loc="upper right")

plt.savefig(os.path.join(sys.path[0], "UG Figure 11A.png"))
