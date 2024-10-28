"""
Purpose:
    Generate a heatmap, for each platform, that displays the mean Spearman's correlation
    between node strength values in naive and PDI-reconstructed networks, averaged
    over 100 versions of the reconstructed network at the specified parameter settings.

Inputs:
    - None. Data is loaded with constants/paths below.

Outputs:
    - .png, .pdf, and .svg files of the heatmap.

Author:
    Matthew DeVerna
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Change the current directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load correlation data. Each row represents a single comparison of node strength
# values (out of 100) between the naive and PDI-reconstructed networks at all parameter settings
print("\nLoading correlation data...")
bs_corrs = pd.read_parquet(
    "/data_volume/cascade_reconstruction/bluesky/networks_stats/strength_differences/bluesky_node_strength_correlations.parquet"
)
mid_corrs = pd.read_parquet(
    "/data_volume/cascade_reconstruction/networks_stats/strength_differences/midterm_node_strength_correlations.parquet"
)

# Calculate the mean correlation between node strength values for each parameter setting
print("Calculating mean correlations...")
bs_mean_corrs = (
    bs_corrs.groupby(["gamma", "alpha"])["spearman_r"]
    .mean()
    .to_frame("mean_correlation")
    .reset_index()
)
mid_mean_corrs = (
    mid_corrs.groupby(["gamma", "alpha"])["spearman_r"]
    .mean()
    .to_frame("mean_correlation")
    .reset_index()
)

print("Creating heatmaps...")
fig, axes = plt.subplots(ncols=2, figsize=(6, 4))

ax1 = axes[0]
ax2 = axes[1]

# Create Bluesky heatmap
heatmap1 = sns.heatmap(
    data=bs_mean_corrs.pivot_table(
        index="gamma", columns="alpha", values="mean_correlation"
    ).T.sort_index(ascending=False),
    vmin=0,
    vmax=1,
    square=True,
    cmap="viridis",
    annot=True,
    ax=ax1,
    cbar=False,  # Disable individual colorbar
)

ax1.set_title("Bluesky")
ax1.set_xlabel(r"$\gamma$", fontsize=12)
ax1.set_ylabel(r"$\alpha$", fontsize=12, labelpad=12, rotation=0, va="center")
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)


# Create Midterm heatmap
heatmap2 = sns.heatmap(
    data=mid_mean_corrs.pivot_table(
        index="gamma", columns="alpha", values="mean_correlation"
    ).T.sort_index(ascending=False),
    vmin=0,
    vmax=1,
    square=True,
    cmap="viridis",
    annot=True,
    ax=ax2,
    cbar=False,  # Disable individual colorbar
)

ax2.set_title("Twitter")
ax2.set_xlabel(r"$\gamma$", fontsize=12)
ax2.set_ylabel(r"$\alpha$", fontsize=12, labelpad=12, rotation=0, va="center")
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

# Add a single colorbar to the bottom of the figure.
# This works because the vmin/vmax values are the same for both heatmaps.
cbar = fig.colorbar(
    heatmap1.get_children()[0],
    ax=axes,
    orientation="horizontal",
    fraction=0.04,
)
cbar.set_label(r"Spearman's $\rho$", fontsize=12)

# Adjust the space between subplots
plt.subplots_adjust(wspace=0.35, bottom=0.3)

# Save the figure in the specified formats
print(f"Saving figures here: {OUTPUT_DIR}")
for ext in ["png", "pdf", "svg"]:
    output_fname = f"correlations_heatmap.{ext}"
    output_fp = os.path.join(OUTPUT_DIR, output_fname)
    fig.savefig(output_fp, bbox_inches="tight", dpi=800)
    print(f"\t- Saved {output_fname}")
