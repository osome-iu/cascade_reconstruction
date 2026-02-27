"""
Purpose:
    This script generates Complementary Cumulative Distribution Function (CCDF) plots
    for various node features (depth, structural virality, and max breadth) of reconstructed
    cascades and time-inferred data.

Inputs:
    - None. Data read via constants/paths defined in the script.

Outputs:
    - CCDF plots for depth, structural virality, and max breadth saved in
    PDF, PNG, and SVG formats.

Author:
    Matthew DeVerna
"""

import os

import glob as glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

mpl.rcParams["font.size"] = 16

# Change the current working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "figures"
METRICS_DIR = "../output/cascade_metrics"
CAS_DIR_BASE = "../output/reconstructed_data"
ALPHA_DIRS = ["alpha_1_1", "alpha_1_5", "alpha_2_0", "alpha_2_5", "alpha_3_0"]
GAMMA_DIRS = ["gamma_0_25", "gamma_0_5", "gamma_0_75"]


def myLogFormat(y, pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = "{{:.{:1d}f}}".format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


print("\nLoading reconstructed data...")
recon_data_df = []
for gamma_dir in GAMMA_DIRS:
    for alpha_dir in ALPHA_DIRS:
        path = os.path.join(
            METRICS_DIR,
            f"cascade_metrics_statistics_{gamma_dir}_{alpha_dir}.parquet",
        )
        recon_data_df.append(pd.read_parquet(path))
recon_data_df = pd.concat(recon_data_df)

print("Loading time-inferred data...")
tid_fname = os.path.join(METRICS_DIR, "time_inferred_diffusion_metrics.parquet")
tid_data = pd.read_parquet(tid_fname)

print("Calculating reconstructed mean values...")
recon_mean_data = (
    recon_data_df.groupby(["cascade_id", "gamma", "alpha"])[
        ["depth", "structural_virality", "max_breadth", "size"]
    ]
    .mean()
    .reset_index()
)


gammas = [0.25, 0.5, 0.75]
alphas = [1.1, 2.0, 3.0]

fig, ax = plt.subplots(
    nrows=3,
    # ncols=3,
    figsize=(5, 10),
    # figsize=(10, 4),
    # sharey=True,
    # sharex=True,
)

color_map = {
    0.25: "#E69F00",
    0.5: "#56B4E9",
    0.75: "#009E73",
    "time-inferred": "purple",
}

line_style_map = {1.1: "solid", 2.0: "dotted", 3.0: "dashed"}

distributions = dict()

for idx_g, gamma in enumerate(gammas):
    for idx_a, alpha in enumerate(alphas):
        print(f"\t- gamma={gamma}, alpha={alpha}...")

        selected_data = recon_mean_data[
            (recon_mean_data.alpha == alpha) & (recon_mean_data.gamma == gamma)
        ]

        for loc, metric in enumerate(["depth", "structural_virality", "max_breadth"]):

            x_vals = sorted(selected_data[metric])
            y_vals = 1 - (np.array(range(len(x_vals))) / len(x_vals))
            distributions[(gamma, alpha, metric)] = x_vals
            ax[loc].plot(
                x_vals,
                y_vals,
                color=color_map[gamma],
                label=r"$\gamma =$" + str(gamma) if idx_a == 0 else "",
                linestyle=line_style_map[alpha],
                linewidth=2.5,
                alpha=0.5,
            )

print("\t- TID depth...")
x_vals = sorted(tid_data["depth"])
y_vals = 1 - (np.array(range(len(x_vals))) / len(x_vals))
distributions[("tid", "tid", "depth")] = x_vals
ax[0].plot(
    x_vals,
    y_vals,
    color=color_map["time-inferred"],
    label="time-inferred",
    linewidth=2.5,
)

print("\t- TID structural virality...")
x_vals = sorted(tid_data["structural_virality"])
y_vals = 1 - (np.array(range(len(x_vals))) / len(x_vals))
distributions[("tid", "tid", "structural_virality")] = x_vals
ax[1].plot(
    x_vals,
    y_vals,
    color=color_map["time-inferred"],
    label="time-inferred",
    linewidth=2.5,
)

print("\t- TID max breadth...")
x_vals = sorted(tid_data["max_breadth"])
y_vals = 1 - (np.array(range(len(x_vals))) / len(x_vals))
distributions[("tid", "tid", "max_breadth")] = x_vals
ax[2].plot(
    x_vals,
    y_vals,
    color=color_map["time-inferred"],
    label="time-inferred",
    linewidth=2.5,
)

# Clean up the plot
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.3)

for idx, a in enumerate(ax):
    a.loglog()

    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

    a.xaxis.set_major_formatter(FuncFormatter(myLogFormat))
    a.yaxis.set_major_formatter(FuncFormatter(myLogFormat))

ax[0].set_xticks([1, 10, 100], [1, 10, 100])
ax[0].set_xlim(0.75, 100)

ax[1].set_xticks([1, 10, 100], [1, 10, 100])
ax[1].set_xlim(0.75, 100)

ax[2].set_xticks([1, 10, 100, 1000, 5000], [1, 10, 100, 1000, 5000])
ax[2].set_xlim(0.75, 5000)

ax[0].set_xlabel("depth")
ax[1].set_xlabel("structural virality")
ax[2].set_xlabel("max breadth")

ax[0].set_ylabel("CCDF\n(prop. of cascades)")
ax[1].set_ylabel("CCDF\n(prop. of cascades)")
ax[2].set_ylabel("CCDF\n(prop. of cascades)")


### BUILD THE LEGEND ###
print("Building the legend...")

# Create custom legend handles
color_handles = [
    Line2D(
        [0],
        [0],
        color=color,
        linewidth=2.5,
        label=r"$\gamma =$" + f"{gamma}" if isinstance(gamma, float) else gamma,
    )
    for gamma, color in color_map.items()
]
style_handles = [
    Line2D(
        [0],
        [0],
        color="grey",
        linewidth=2.5,
        linestyle=line_style_map[alpha],
        label=r"$\alpha =$" + f"{alpha}",
    )
    for alpha in alphas
]

# Create the color legend above the panels
color_legend = fig.legend(
    handles=color_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    ncol=len(color_map.keys()),
    fontsize=15,
    frameon=False,
)

# Create the style legend above the color legend
style_legend = fig.legend(
    handles=style_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(alphas),
    fontsize=15,
    frameon=False,
)

ax[0].annotate("(a)", xy=(-0.3, 0.99), xycoords=ax[0].transAxes)
ax[1].annotate("(b)", xy=(-0.3, 0.99), xycoords=ax[1].transAxes)
ax[1].annotate("(c)", xy=(-0.3, 0.99), xycoords=ax[2].transAxes)


# Define the file name and extensions
file_name = "depth_sv_breadth"
extensions = ["pdf", "png", "svg"]

# Save the figure in different formats using a loop
for ext in extensions:
    output_path = os.path.join(OUTPUT_DIR, f"{file_name}.{ext}")
    fig.savefig(output_path, dpi=800, bbox_inches="tight")
    print(f"- Created: {output_path}")
