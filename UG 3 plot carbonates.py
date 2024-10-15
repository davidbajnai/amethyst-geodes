# This code is used to:
# Plot the geolde calcite data in triple oxygen isotope space

# INPUT:  UG Table S4.csv (carbonate data)
#         UG fluid model late calcite.csv (best-fit compositions)
#         meteoric_water.csv (modern meteoric water data)
#         UG Table S2.csv (measured geode water data)
#         evaporation.csv (simple evaporation model based on Voigt et al., 2021)

# OUTPUT: UG Figure 11B.png

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


# Functions that make life easier

def a18_cc(T):

    # Hayles et al. (2018) - calcite
    # B_calcite = 7.027321E+14 / T**7 + -1.633009E+13 / T**6 + 1.463936E+11 / T**5 + -5.417531E+08 / T**4 + -4.495755E+05 / T**3  + 1.307870E+04 / T**2 + -5.393675E-01 / T + 1.331245E-04
    # B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    # return np.exp(B_calcite) / np.exp(B_water)

    # Alternative equations
    # return np.exp((2.84 * 10**6 / T**2 - 2.96) / 1000) # Wostbrock et al. (2020) – calcite
    # return np.exp((17.88 * 1000 / T - 31.14) / 1000)   # Kim et al. (2007) – aragonite
    return np.exp((17.57 * 1000 / T - 29.13) / 1000)     # Daeron et al. (2019) – calcite
    # return 0.0201 * (1000 / T) + 0.9642                # Guo and Zhou (2019) – aragonite


def theta_cc(T):
    # Hayles et al. (2018) - calcite
    K_calcite = 1.019124E+09 / T**5 + -2.117501E+07 / T**4 + 1.686453E+05 / T**3 + -5.784679E+02 / T**2 + 1.489666E-01 / T + 0.5304852
    B_calcite = 7.027321E+14 / T**7 + -1.633009E+13 / T**6 + 1.463936E+11 / T**5 + -5.417531E+08 / T**4 + -4.495755E+05 / T**3  + 1.307870E+04 / T**2 + -5.393675E-01 / T + 1.331245E-04
    K_water = 7.625734E+06 / T**5 + 1.216102E+06 / T**4 + -2.135774E+04 / T**3 + 1.323782E+02 / T**2 + -4.931630E-01 / T + 0.5306551
    B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    a18 = np.exp(B_calcite) / np.exp(B_water)
    return K_calcite + (K_calcite-K_water) * (B_water / np.log(a18))

    # Alternative equations
    # return -1.39 / T + 0.5305                 # Wostbrock et al. (2020) – calcite
    # return 59.1047/T**2 + -1.4089/T + 0.5297  # Guo and Zhou (2019) – aragonite
    # return -1.53 / T + 0.5305                 # Wostbrock et al. (2020) – aragonite


def a17_cc(T):
    return a18_cc(T)**theta_cc(T)


def d18O_cc(equilibrium_temperatures, d18Ow):
    return a18_cc(equilibrium_temperatures) * (d18Ow+1000) - 1000


def d17O_cc(equilibrium_temperatures, d18Ow):
    return a17_cc(equilibrium_temperatures) * (d18Ow+1000) - 1000


def plot_calcite_equilibrium(Dp17Ow, d18Ow, Tmin, Tmax, ax, fluid_name="precipitating fluid", color="k", highlight=True):

    d17Ow = d17O(d18Ow, Dp17Ow)

    ax.scatter(prime(d18Ow), Dp17O(d17Ow, d18Ow),
            marker="X", fc=color, ec="k", zorder=10, label=fluid_name)

    # equilibrium, entire T range
    toInf = np.arange(0, 330, 1) + 273.15
    d18O_mineral = d18O_cc(toInf, d18Ow)
    d17O_mineral = d17O_cc(toInf, d17Ow)
    mineral_equilibrium = np.array(
        [d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), toInf]).T
    ax.plot(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
            ":", c=color, zorder=3)

    # equilibrium, highlight range
    equilibrium_temperatures = np.arange(Tmin, Tmax, 0.5) + 273.15
    colors = np.linspace(0, 1, len(equilibrium_temperatures))
    d18O_mineral = d18O_cc(equilibrium_temperatures, d18Ow)
    d17O_mineral = d17O_cc(equilibrium_temperatures, d17Ow)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(
        d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
    if highlight == True:
        ax.scatter(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
                   marker=".", c=colors, cmap='coolwarm', zorder=3)

    # equilibrium, highlight range, marker every 10 °C
    equilibrium_temperatures = np.arange(Tmin, Tmax+1, 10) + 273.15
    d18O_mineral = d18O_cc(equilibrium_temperatures, d18Ow)
    d17O_mineral = d17O_cc(equilibrium_temperatures, d17Ow)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(
        d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
    if highlight == True:
        ax.scatter(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
                   s=15, marker="o", fc="white", ec=color, zorder=3)

    # Return equilibrium data as a dataframe
    equilibrium_df = pd.DataFrame(mineral_equilibrium)
    equilibrium_df[2] = equilibrium_df[2]-273.15
    equilibrium_df = equilibrium_df.rename(
        columns={0: 'd18O', 1: 'Dp17O', 2: 'temperature'})
    return equilibrium_df


# Read in data
df_cal = pd.read_csv(os.path.join(sys.path[0], "UG Table S4.csv"))
meteoricwdf = pd.read_csv(os.path.join(sys.path[0], "meteoric_water.csv"))
UYFARwdf = pd.read_csv(os.path.join(sys.path[0], "UG Table S2.csv"))
df_modeled_fluids = pd.read_csv(os.path.join(sys.path[0], "UG fluid model late calcite.csv"))
evap = pd.read_csv(os.path.join(sys.path[0], "evaporation.csv"))

# Start plotting
fig, ax = plt.subplots()

# Plot meteoric waters
ax.scatter(prime(meteoricwdf["d18O"]), Dp17O(meteoricwdf["d17O"], meteoricwdf["d18O"]),
           marker="+", c="#cacaca", zorder=-10,
           label="modern meteoric waters")

# Plot measured UYFAR waters
confidence_ellipse(prime(UYFARwdf["d18O"]), UYFARwdf["Dp17O"], ax, n_std=2,
                   ec="#005752", zorder=10, lw=1)

dfeq = plot_calcite_equilibrium(UYFARwdf["Dp17O"].mean(), UYFARwdf["d18O"].mean(), 0, 80,
                                ax, fluid_name="measured geode water", color="#005752", highlight=False)
    

confidence_ellipse(prime(df_modeled_fluids["d18Ow"]), df_modeled_fluids["Dp17Ow"], ax, n_std=1 * 2,
                   ec="#63A615", zorder=1, lw=1)
fluid2_equi = plot_calcite_equilibrium(df_modeled_fluids["Dp17Ow"].mean(), df_modeled_fluids["d18Ow"].mean(), 10, 60,
                                       ax, fluid_name="modeled fluid: late cc", color="#63A615")

# annotate the firs and last point in dfeq
ax.annotate(str(round(fluid2_equi.iloc[0, 2])) + " °C", (prime(fluid2_equi.iloc[0, 0]), fluid2_equi.iloc[0, 1]),
            xytext=(prime(fluid2_equi.iloc[0, 0])+5, fluid2_equi.iloc[0, 1]),
            arrowprops=dict(arrowstyle="-|>", color="#4F4B41"))
ax.annotate(str(round(fluid2_equi.iloc[-1, 2])) + " °C", (prime(fluid2_equi.iloc[-1, 0]), fluid2_equi.iloc[-1, 1]),
            xytext=(prime(fluid2_equi.iloc[-1, 0])-5, fluid2_equi.iloc[-1, 1]), ha="right",
            arrowprops=dict(arrowstyle="-|>", color="#4F4B41"))


# Empty dotted line for legend
ax.plot([], [], ":", c="k", label="equilibrium CaCO$_3$")

# Plot carbonate samples
cat1 = df_cal["Type"].unique()
markers = dict(zip(cat1, ["D", "s"]))
cat2 = df_cal["Type"].unique()
colors = dict(zip(cat2, ["#1455C0", "#B4D5F6"]))

for cat in cat1:
    for dog in cat2:
        dat = df_cal[(df_cal["Type"] == cat) & (df_cal["Type"] == dog)]
        if len(dat) > 0:
            x = prime(dat["d18O_AC"])
            y = dat["Dp17O_AC"]
            ax.scatter(x, y, marker=markers[cat],
                       color=colors[dog], label=f"{dog} calcite", edgecolors="k", zorder=10)


# Add arrows
A = (dfeq.iloc[3, 0], dfeq.iloc[3, 1])
ax.scatter(prime(A[0]), A[1],
           marker="o", fc="w", ec="#005752", zorder=10, s=20)
ax.annotate(str(round(dfeq.iloc[3, 2])) + " °C", (prime(dfeq.iloc[3, 0]), dfeq.iloc[3, 1]),
            xytext=(prime(dfeq.iloc[3, 0]+5), dfeq.iloc[3, 1]),
            arrowprops=dict(arrowstyle="-|>", color="#4F4B41"))

# CO2 degassing
theta = 0.534
shift_d18O = 6
B = (A[0]+shift_d18O, apply_theta(A[0], A[1], shift_d18O=shift_d18O, theta=theta))
ax.annotate("",
            xy=(prime(A[0]), A[1]),
            xytext=(prime(B[0]), B[1]),
            ha="center", va="center", zorder=-1,
            arrowprops=dict(arrowstyle="<|-", color="#FF7A00", lw=3))
ax.text(B[0], B[1], "CO$_2$\ndegassing",
        ha="center", va="bottom", color="#FF7A00")

# CO2 absorption
theta = 0.532
shift_d18O = -5
B = (A[0]+shift_d18O, apply_theta(A[0], A[1], shift_d18O=shift_d18O, theta=theta))
ax.annotate("",
            xy=(prime(A[0]), A[1]),
            xytext=(prime(B[0]), B[1]),
            ha="center", va="center", zorder=-1,
            arrowprops=dict(arrowstyle="<|-", color="#4D0820", lw=3))
ax.text(B[0], B[1], "CO$_2$\nabsorbtion",
        ha="right", va="center", color="#4D0820",
        bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))

ax.annotate("", xy=(prime(A[0]), A[1]),
            xytext=(prime(dfeq.iloc[len(dfeq)-1, 0]),
                    dfeq.iloc[len(dfeq)-1, 1]),
            ha="center", va="bottom", color="#EC0016",
            bbox=dict(fc='white', ec='none', pad=0.5),
            arrowprops=dict(arrowstyle="<|-", color="#EC0016", lw=3))
ax.text(prime(dfeq.iloc[len(dfeq)-1, 0]), dfeq.iloc[len(dfeq)-1, 1],
        "higher\ngrowth T",
        bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8),
        ha="right", va="bottom", color="#EC0016")

ax.annotate("", xy=(prime(A[0]), A[1]),
            xytext=(prime(dfeq.iloc[0, 0]), dfeq.iloc[0, 1]),
            ha="center", va="bottom", color="#FFBB00",
            bbox=dict(fc='white', ec='none', pad=0.5),
            arrowprops=dict(arrowstyle="<|-", color="#FFBB00", lw=3))
ax.text(prime(dfeq.iloc[0, 0])+1, dfeq.iloc[0, 1],
        "lower\ngrowth T",
        bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8),
        ha="left", va="bottom", color="#FFBB00")


# Simple evaporation model (from Voigt et al., 2021)
# Residual water
ax.plot(prime(evap["d18O"][:-2]), evap["Dp17O"]
        [:-2], c="#3CB5AE", zorder=-2, lw=3)
ax.annotate("", xy=(prime(evap["d18O"].iloc[-3]), evap["Dp17O"].iloc[-3]),
            xytext=(prime(evap["d18O"].iloc[-1]), evap["Dp17O"].iloc[-1]),
            ha="center", va="center", color="#3CB5AE", zorder=-1,
            arrowprops=dict(arrowstyle="<|-", color="#3CB5AE", lw=3))
ax.text(5, -20, "evaporation",
        ha="left", va="center", color="#3CB5AE",
        bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))


# MORB
ax.scatter(prime(5.7), -51,
           marker="*", fc="k", ec = "w", s = 60,
           label="MORB")
MORB_mix = mix_d17O(d18O_A=UYFARwdf["d18O"].mean(), D17O_A=UYFARwdf["Dp17O"].mean(),
                    d18O_B = 5.7, D17O_B = -51,
                    step=100)
ax.plot(prime(MORB_mix["mix_d18O"][:-10]), MORB_mix["mix_Dp17O"][:-10], c="#2A7230", zorder=0, lw=3)
ax.annotate("",
            xy=(prime(MORB_mix["mix_d18O"].iloc[-20]), MORB_mix["mix_Dp17O"].iloc[-20]),
            xytext=(prime(MORB_mix["mix_d18O"].iloc[-1]), MORB_mix["mix_Dp17O"].iloc[-1]),
            ha="center", va="bottom", color="#154A26",
            bbox=dict(fc='white', ec='none', pad=0.5),
            arrowprops=dict(arrowstyle="<|-", color="#2A7230", lw=3))
ax.text(-3, 0, "exchange with\nbasalt",
        ha="right", va="center", color="#2A7230",
        bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))


# Plot error bar
ax.errorbar(-20, -120,
            xerr=df_cal["d18O_error"].mean(),
            yerr=df_cal["Dp17O_error"].mean(), fmt="none",
            color="gray", elinewidth=0.6, zorder=0)
ax.text(-19.5, -120,
         "mean error of\ncarbonate $\Delta\prime^{17}$O",
         fontsize=6, va="center", ha="left", color="gray")


ax.text(0.02, 0.98, "b", fontsize=14, fontweight="bold",
        va="top", ha="left", transform=ax.transAxes,
        bbox=dict(fc="w", pad=0.1, ec="none", alpha=0.8))

ax.set_xlabel("$\delta\prime^{18}$O (‰, VSMOW)")
ax.set_ylabel("$\Delta\prime^{17}$O (ppm)")

ax.set_ylim(-155, 105)
ax.set_xlim(-25, 45)
ax.legend(loc="upper right")

plt.savefig(os.path.join(sys.path[0], "UG Figure 11B.png"))
