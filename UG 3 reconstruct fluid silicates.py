# This code is used to:
# Model ambient water oxygen isotope compositions based on the amethyst data

# INPUT:  UG Table S3.csv (quartz data)

# OUTPUT: UG fluid model amethyst.csv (modeled compositions)

# >>>>>>>>>

# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from functions import *

# Plot parameters
plt.rcParams.update({"font.size": 6})
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["patch.linewidth"] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'


# Functions that make life easier

def a18_qz(T):

    # Sharp et al. (2016) - Eq. 9
    return np.exp((4.28 * 10**6 / T**2 - 3.5 * 10**3 / T) / 1000)


def theta_qz(T):
    
    # Sharp et al. (2016) - Eq. 10
    return -1.85 / T + 0.5305


def a17_qz(T):
    return a18_qz(T)**theta_qz(T)


def d18O_qz(equilibrium_temperatures, d18Ow):
    return a18_qz(equilibrium_temperatures) * (d18Ow+1000) - 1000


def d17O_qz(equilibrium_temperatures, d17Ow):
    return a17_qz(equilibrium_temperatures) * (d17Ow+1000) - 1000


# Read quartz data from CSV file
df = pd.read_csv(os.path.join(sys.path[0], "UG Table S3.csv"))

# Filter data for amethyst
df = df[df["Type"] == "amethyst"]
df["Dp17O"] = Dp17O(df["d17O"], df["d18O"])
print(df)

# Plot parameters
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

# First plot and model

# Plot the samples
ax1.scatter(prime(df["d18O"]), Dp17O(df["d17O"], df["d18O"]),
            marker="o", fc="#1455C0", ec="w", lw=0.5,
            zorder=10, label="samples")

# Create an empty dataframe to store the modeled values
modeldf = pd.DataFrame(columns=["d18Ow", "d17Ow", "Dp17Ow", "sum_distance", "avg_temperature", "min_temperature", "max_temperature"])

# Loop over a range of d18Ow and Dp17Ow values
T_min, T_max = 0, 300  # temperature range for the equilibrium calculations
d18Ow_min, d18Ow_max, d18Ow_step = -16, -4, 0.2
Dp17Ow_min, Dp17Ow_max, Dp17Ow_step = 10, 70, 2

model_length = ((d18Ow_max-d18Ow_min)/d18Ow_step) * ((Dp17Ow_max-Dp17Ow_min)/Dp17Ow_step)
print(f"Modeling {model_length:.0f} fluids")

for d18Ow in tqdm(np.arange(d18Ow_min, d18Ow_max, d18Ow_step)):
    for Dp17Ow in np.arange(Dp17Ow_min, Dp17Ow_max, Dp17Ow_step):

        d17Ow = d17O(d18Ow, Dp17Ow)

        # Calculate equilibrium points between 0 °C and 300 °C with 1 degree resolution
        equilibrium_temperatures = np.arange(T_min, T_max+1, 1) + 273.15
        d18O_mineral = d18O_qz(equilibrium_temperatures, d18Ow)
        d17O_mineral = d17O_qz(equilibrium_temperatures, d17Ow)
        mineral_equilibrium = np.array([d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), equilibrium_temperatures]).T

        ax1.plot(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
                 ls="solid", color="grey", alpha=0.3, label="quartz equilibrium", zorder=-1)
        ax1.scatter(prime(d18Ow), Dp17O(d17Ow, d18Ow),
                    marker="d", fc="w", ec="k", label=f"model fluids ($\\mathit{{N}}$ = {model_length:.0f})")

        data = []
        for i, row in df.iterrows():
            A = np.array([row["d18O"], row["Dp17O"]])
            distances = np.linalg.norm(mineral_equilibrium[:, :2] - A, axis=1)
            mindist = np.min(distances)
            closest_index = np.argmin(distances)
            closest_point = mineral_equilibrium[closest_index]
            tempera = closest_point[2]
            ax1.plot([prime(A[0]), prime(closest_point[0])], [A[1], closest_point[1]],
                     color="#63A615", ls="-", linewidth=0.4, alpha=0.3,
                     label="distance to closest equi. point")
            data.append({"distances": mindist, "temperatures": tempera})

        modeldfa = pd.DataFrame(data)

        modeldf = modeldf.dropna(axis=1, how='all')
        modeldf = pd.concat([modeldf, pd.DataFrame({"d17Ow": [d17Ow], "Dp17Ow": [Dp17Ow], "d18Ow": [d18Ow],
                                                    "sum_distance": [np.sum(modeldfa["distances"])],
                                                    "avg_temperature": [np.mean(modeldfa["temperatures"])-273.15],
                                                    "min_temperature": [np.min(modeldfa["temperatures"])-273.15],
                                                    "max_temperature": [np.max(modeldfa["temperatures"])-273.15]}
                                                   )
                             ], ignore_index=True)
        modeldfa = []

print("Modeling complete")

handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), loc="upper right")

# Plot axes
ax1.text(0.02, 0.98, "a", fontsize=14, fontweight="bold",
         va="top", ha="left", transform=ax1.transAxes)
ax1.set_xlabel("$\delta\prime^{18}$O (‰, VSMOW)")
ax1.set_ylabel("$\Delta\prime^{17}$O (ppm)")
ax1.set_ylim(-155, 105)
ax1.set_xlim(-25, 55)

print("Plot 1 complete")

# Second plot

ax2.scatter(modeldf["sum_distance"], modeldf["avg_temperature"],
            marker="d", fc="w", ec="k", label=f"model fluids ($\\mathit{{N}}$ = {model_length:.0f})")

# Define the cut-off values
T_cut_lower = 19
T_cut_upper = 30
Dist_cut = modeldf['sum_distance'].quantile(0.10)

# Display the cut-off values
ax2.axhline(y=T_cut_upper, color="#EC0016", linestyle="-", zorder=3, label = "cut-off")
ax2.axhline(y=T_cut_lower, color="#EC0016", linestyle="-", zorder=3)
ax2.axvline(x=Dist_cut, color="#EC0016", linestyle="-", zorder=3)

xmin, xmax = ax2.get_xlim()
ax2.text(xmax, T_cut_upper, str(T_cut_upper)+" °C",
         color="#EC0016", va="bottom", ha="right",
         bbox=dict(fc="w", pad=0.1, ec="none", alpha=0.8))
ax2.text(xmax, T_cut_lower, str(T_cut_lower)+" °C",
         color="#EC0016", va="top", ha="right",
         bbox=dict(fc="w", pad=0.1, ec="none", alpha=0.8))

# Filter the modeled fluids
modeldf = modeldf[modeldf["sum_distance"] <= Dist_cut]
modeldf = modeldf[modeldf["avg_temperature"] >= T_cut_lower]
modeldf = modeldf[modeldf["avg_temperature"] <= T_cut_upper]

ax2.scatter((modeldf["sum_distance"]), modeldf["avg_temperature"],
            marker="d", fc="#EC0016", ec="#EC0016", label=f"best-fit fluids ($\\mathit{{N}}$ = {modeldf.shape[0]:.0f})")

ax2.legend(loc="upper right")
ax2.set_xlabel("sum of distances")
ax2.set_ylabel("average temperature (°C)")
ax2.text(0.02, 0.98, "b", fontsize=14, fontweight="bold",
         va="top", ha="left", transform=ax2.transAxes,
         bbox=dict(fc="w", pad=0.1, ec="none", alpha=0.8))

print("Plot 2 complete")


# Third plot

ax3.scatter(prime(df["d18O"]), Dp17O(df["d17O"], df["d18O"]),
            marker="o", fc="#1455C0", ec="w", lw=0.5,
            zorder=10, label="samples")

# Rectangle for the fluid range considered in the model
ax3.add_patch(Rectangle((d18Ow_min, Dp17Ow_min), d18Ow_max-d18Ow_min,
                        (Dp17Ow_max-Dp17Ow_min),
                        fc="#DDDED6", ec=None, zorder=-10))

for i, row in modeldf.iterrows():
    ax3.scatter(prime(row["d18Ow"]), Dp17O(row["d17Ow"], row["d18Ow"]),
                marker="d", fc="#EC0016", ec="#EC0016", label="best-fit fluids", zorder=2)

    # Plot the equilibrium line between 0–1000°C
    equilibrium_temperatures = np.arange(T_min,T_max+1,1) + 273.15
    d18O_mineral = d18O_qz(equilibrium_temperatures, row["d18Ow"])
    d17O_mineral = d17O_qz(equilibrium_temperatures, row["d17Ow"])
    ax3.scatter(prime(d18O_mineral), Dp17O(d17O_mineral, d18O_mineral),
            s=0.5, marker="o", fc = "k", ec="none", alpha=0.3,
            label=f"equilibrium ({np.min(equilibrium_temperatures-273.15):.0f}–{np.max(equilibrium_temperatures-273.15):.0f} °C)")
confidence_ellipse(prime(modeldf["d18Ow"]), modeldf["Dp17Ow"], ax3,
                   ec="k", zorder=10, label="$\pm$1$\sigma$ CI, modeled fluids")


handles, labels = ax3.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax3.legend(by_label.values(), by_label.keys(), loc="upper right")

ax3.set_xlabel("$\delta\prime^{18}$O (‰, VSMOW)")
ax3.set_ylabel("$\Delta\prime^{17}$O (ppm)")
ax3.text(0.02, 0.98, "c", fontsize=14, fontweight="bold",
         va="top", ha="left", transform=ax3.transAxes,
         bbox=dict(fc="w", pad=0.1, ec="none", alpha=0.8))
ax3.set_ylim(-155, 105)
ax3.set_xlim(-25, 55)

print("Plot 3 complete")


# Fourth plot

ax4.scatter(prime(df["d18O"]), Dp17O(df["d17O"], df["d18O"]),
            marker="o", fc="#1455C0", ec="w", lw=0.5,
            zorder=10, label="samples")

# Rectangle for the fluid range considered in the model
ax4.add_patch(Rectangle((d18Ow_min, Dp17Ow_min), d18Ow_max-d18Ow_min,
                        (Dp17Ow_max-Dp17Ow_min),
                        fc="#DDDED6", ec=None, zorder=-10))
                    
# model water
mean_d18Ow = modeldf["d18Ow"].mean()
sd_d18Ow = modeldf["d18Ow"].std()
mean_d17Ow = np.mean(modeldf["d17Ow"])
mean_Dp17Ow = np.mean(modeldf["Dp17Ow"])
sd_Dp17Ow = np.std(modeldf["Dp17Ow"])

ax4.errorbar(prime(mean_d18Ow), mean_Dp17Ow,
             xerr=sd_d18Ow,
             yerr=sd_Dp17Ow,
             ecolor="k", marker="d", mfc="w", mec="k", ls="none",
             label="modeled fluid")
confidence_ellipse(prime(modeldf["d18Ow"]), modeldf["Dp17Ow"], ax4,
                   ec="k", zorder=10, label="$\pm$1$\sigma$ CI, modeled fluids")

ax4.text(0.05, 0.2,
         "modeled best-fit fluid composition:\n" +
         f"$\delta^{{18}}$O: {mean_d18Ow:.1f}$\pm${sd_d18Ow:.1f}‰\n$\Delta\prime^{{17}}$O: {mean_Dp17Ow:.0f}$\pm${sd_Dp17Ow:.0f} ppm",
         color="k", ha="left", va="top", transform=ax4.transAxes)

equilibrium_temperatures = np.arange(T_min, T_max+1, 1) + 273.15
d18O_mineral = d18O_qz(equilibrium_temperatures, mean_d18Ow)
d17O_mineral = d17O_qz(equilibrium_temperatures, mean_d17Ow)
ax4.plot(prime(d18O_mineral), Dp17O(d17O_mineral, d18O_mineral),
         ":", lw=0.5, zorder=1, color="k",
         label=f"equilibrium ({np.min(equilibrium_temperatures-273.15):.0f}–{np.max(equilibrium_temperatures-273.15):.0f} °C)")


equilibrium_temperatures = np.arange(10, 81, 1) + 273.15
d18O_mineral = d18O_qz(equilibrium_temperatures, mean_d18Ow)
d17O_mineral = d17O_qz(equilibrium_temperatures, mean_d17Ow)
ax4.plot(prime(d18O_mineral), Dp17O(d17O_mineral, d18O_mineral),
         "-", lw=2, mec="white", zorder=1, color="k",
         label=f"equilibrium ({np.min(equilibrium_temperatures-273.15):.0f}–{np.max(equilibrium_temperatures-273.15):.0f} °C)")


ax4.set_xlabel("$\delta\prime^{18}$O (‰, VSMOW)")
ax4.set_ylabel("$\Delta\prime^{17}$O (ppm)")
ax4.set_ylim(-155, 105)
ax4.set_xlim(-25, 55)
ax4.legend(loc="upper right")
ax4.text(0.02, 0.98, "d", fontsize=14, fontweight="bold",
         va="top", ha="left", transform=ax4.transAxes)

print("Plot 4 complete")

plt.savefig(os.path.join(sys.path[0], "UG Figure S6.png"))
print("Figure saved")

modeldf = round(modeldf, 4)
modeldf.to_csv(os.path.join(sys.path[0], "UG fluid model amethyst.csv"), index=False)
print("Modeled fluid values exported to CSV")