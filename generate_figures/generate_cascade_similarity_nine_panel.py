"""
Purpose:
    This script generates a nine-panel figure illustrating cascade similarity metrics. 

Input:
    - None. Data loaded with constants defined in the script.

Output:
    - Multiple versions of the figure (PNG, PDF, and SVG) in the OUTPUT_DIR directory.

Author:
    Matthew DeVerna
"""

import os

import glob as glob
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

mpl.rcParams["font.size"] = 12

# Ensures the relative paths work when running the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SIM_DIR = "../output/cascade_similarity_metrics"
OUTPUT_DIR = "figures"


def myLogFormat(y, pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = "{{:.{:1d}f}}".format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


print("\nLoading data...")
files = sorted(glob.glob(os.path.join(SIM_DIR, "*")))
dfs = []
for file in files:
    if "pdi" in file:
        recon_type = "pdi"
    else:
        recon_type = "tid"

    df = pd.read_parquet(file)
    df["recon_type"] = recon_type
    dfs.append(df)
sim_mets_df = pd.concat(dfs)


# Plot the data
fig, ax = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(10, 10),
    sharex=True,
    sharey=True,
)

row_ix = 0
for gamma in [0.25, 0.5, 0.75]:

    col_ix = 0

    for alpha in [1.1, 2.0, 3.0]:
        selected_data = sim_mets_df[
            (sim_mets_df.alpha == alpha) & (sim_mets_df.gamma == gamma)
        ]

        ax[row_ix][col_ix].grid(axis="y", zorder=0)

        sns.regplot(
            data=selected_data[selected_data["recon_type"] == "pdi"],
            x="size",
            y="jaccard_mean",
            lowess=True,
            x_bins=500,
            n_boot=1000,
            scatter_kws={"s": 10, "marker": "x", "alpha": 0.5},
            label="PDI vs. PDI",
            color="k",
            ax=ax[row_ix][col_ix],
        )

        sns.regplot(
            data=selected_data[selected_data["recon_type"] == "tid"],
            x="size",
            y="jaccard_mean",
            lowess=True,
            x_bins=500,
            n_boot=1000,
            scatter_kws={"s": 10, "marker": "x", "alpha": 0.5},
            label="PDI vs. TID",
            color="tomato",
            ax=ax[row_ix][col_ix],
        )

        # Set x- and y-axis labels later
        ax[row_ix][col_ix].set_ylabel("")
        ax[row_ix][col_ix].set_xlabel("")
        ax[row_ix][col_ix].set_xscale("log")

        ax[row_ix][col_ix].set_title(
            label=r"$\gamma =$" + f"{gamma}\n" + r"$\alpha =$" + f"{alpha}",
            fontsize=11,
            y=0.8,
        )

        col_ix += 1

    ax[row_ix][0].set_ylim(0, 1)
    ax[row_ix][0].set_ylabel("cascade similarity")

    row_ix += 1

ax[1][2].legend(frameon=True, loc="center right", fontsize=10)

ax[2][0].set_xlabel("cascade size")
ax[2][1].set_xlabel("cascade size")
ax[2][2].set_xlabel("cascade size")

ax[2][0].xaxis.set_major_formatter(FuncFormatter(myLogFormat))

sns.despine()

plt.subplots_adjust(hspace=0.1, wspace=0.2)


# Save the figure in the specified formats
print(f"Saving figures here: {OUTPUT_DIR}")
for ext in ["png", "pdf", "svg"]:
    output_fname = f"cascade_sim_multi_panel.{ext}"
    output_fp = os.path.join(OUTPUT_DIR, output_fname)
    fig.savefig(output_fp, bbox_inches="tight", dpi=800)
    print(f"\t- Saved {output_fname}")
