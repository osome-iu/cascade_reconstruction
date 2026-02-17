"""
Purpose:
    Generate a six-panel figure illustrating the effect of reconstruction on
    node influence for both Bluesky and Twitter data, comparing top k% influentials.

Inputs:
    - None

Outputs:
    - The figure in three forms: .png, .pdf, and .svg


Author:
    - Matthew DeVerna
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

mpl.rcParams["font.size"] = 14
mpl.rcParams["xtick.labelsize"] = 12  # Set x-tick label size
mpl.rcParams["ytick.labelsize"] = 12  # Set y-tick label size
twitter_color = "#0f1419"
bsky_color = "#1185fe"

# Change the current working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the output directory
OUTPUT_DIR = "figures"


def custom_formatter(x, pos):
    """
    Convert exponent values to whole numbers with commas.
    :param x: the tick value
    :param pos: the position of the tick (required by FuncFormatter but not used)
    :return: formatted tick label as a string
    """
    s = f"{x:,.0f}"
    return s


# Set paths
# NOTE: These will need to be updated to the correct paths for replication based on where you save the data.
cascade_reconstruction_dir = "/data_volume/cascade_reconstruction/"
bs_jaccard_file = os.path.join(
    cascade_reconstruction_dir,
    "bluesky/networks_stats/jaccard_coefficients_pdi_vs_naive.parquet",
)
mid_jaccard_file = os.path.join(
    cascade_reconstruction_dir,
    "networks_stats/jaccard_coefficients_pdi_vs_naive.parquet",
)
mid_data_dir = os.path.join(
    cascade_reconstruction_dir, "networks_stats/strength_differences/"
)
bs_data_dir = os.path.join(
    cascade_reconstruction_dir, "bluesky/networks_stats/strength_differences/"
)

# Load the top influencer comparisons file for each platform and select
# a specific parameter setting
bsky_df = pd.read_parquet(bs_jaccard_file)
bsky_df = bsky_df[
    (bsky_df.metric == "strength")
    & (bsky_df.gamma == 0.25)
    & (bsky_df.alpha == 3.0)
    & (bsky_df.k < 101)
]

mid_df = pd.read_parquet(mid_jaccard_file)
mid_df = mid_df[
    (mid_df.metric == "strength")
    & (mid_df.gamma == 0.25)
    & (mid_df.alpha == 3.0)
    & (mid_df.k < 101)
]

# Gather the strength change files
mid_files = os.listdir(mid_data_dir)
bs_files = os.listdir(bs_data_dir)

# Remove, select, and load the average change files for each platform at the same parameter setting
one_mid_file = mid_files.pop(
    mid_files.index("mean_strength_change_gamma_0.25_alpha_3.0.parquet")
)
one_bs_file = bs_files.pop(
    bs_files.index("mean_strength_change_gamma_0.25_alpha_3.0.parquet")
)
mid_strength_change = pd.read_parquet(os.path.join(mid_data_dir, one_mid_file))
bs_strength_change = pd.read_parquet(os.path.join(bs_data_dir, one_bs_file))

# Load a specific strength change files for each platform
raw_mid_file = mid_files.pop(
    mid_files.index("strength_change_gamma_0.25_alpha_3.0.parquet")
)
raw_bs_file = bs_files.pop(
    bs_files.index("strength_change_gamma_0.25_alpha_3.0.parquet")
)
raw_mid_strength_change = pd.read_parquet(os.path.join(mid_data_dir, raw_mid_file))
raw_bs_strength_change = pd.read_parquet(os.path.join(bs_data_dir, raw_bs_file))

# Note that the above files will have 100 comparisons at each alpha and gamma value
# so we select only the first version as an example.
raw_bs_strength_change = raw_bs_strength_change[raw_bs_strength_change.net_v == 1][
    ["user_id", "strength_reconstruct", "strength_naive"]
]

raw_mid_strength_change = raw_mid_strength_change[raw_mid_strength_change.net_v == 1][
    ["user_id", "strength_reconstruct", "strength_naive"]
]

# Count the number of instances of each value
mid_strength_change_count = (
    mid_strength_change.groupby(
        ["strength_naive", "mean_strength_diff_recon_minus_naive"]
    )["user_id"]
    .count()
    .to_frame("count")
    .reset_index()
)
bs_strength_change_count = (
    bs_strength_change.groupby(
        ["strength_naive", "mean_strength_diff_recon_minus_naive"]
    )["user_id"]
    .count()
    .to_frame("count")
    .reset_index()
)


raw_bs_strength_change = (
    raw_bs_strength_change.groupby(["strength_naive", "strength_reconstruct"])[
        "user_id"
    ]
    .count()
    .to_frame("count")
    .reset_index()
)
raw_mid_strength_change = (
    raw_mid_strength_change.groupby(["strength_naive", "strength_reconstruct"])[
        "user_id"
    ]
    .count()
    .to_frame("count")
    .reset_index()
)

# Clip count values for better visualization
ds = [
    raw_bs_strength_change,
    raw_mid_strength_change,
    mid_strength_change_count,
    bs_strength_change_count,
]
for d in ds:
    d["count_clipped"] = d["count"].clip(upper=500)

# Calculate the median strength chance for each platform
bs_median_strength_diff = (
    bs_strength_change_count.groupby("strength_naive")[
        "mean_strength_diff_recon_minus_naive"
    ]
    .median()
    .to_frame("median_strength_diff")
    .reset_index()
)

bs_median_strength_diff = bs_median_strength_diff.sort_values("strength_naive")

mid_median_strength_diff = (
    mid_strength_change_count.groupby("strength_naive")[
        "mean_strength_diff_recon_minus_naive"
    ]
    .median()
    .to_frame("median_strength_diff")
    .reset_index()
)

mid_median_strength_diff = mid_median_strength_diff.sort_values("strength_naive")


############################################
####### BEGIN PLOTTING FIGURE #######
############################################


# Create a figure
fig = plt.figure(figsize=(9.5, 13))

# Define the grid layout
gs = GridSpec(nrows=3, ncols=2, figure=fig)

# Create subplots
ax1 = fig.add_subplot(gs[0, 0])  # Bluesky scatter (top-left)
ax2 = fig.add_subplot(gs[1, 0])  # Bluesky avg change (middle-left)
ax3 = fig.add_subplot(gs[2, 0])  # Bluesky top k% (bottom-left)
ax4 = fig.add_subplot(gs[0, 1])  # Twitter scatter (top-right)
ax5 = fig.add_subplot(gs[1, 1])  # Twitter avg change (middle-right)
ax6 = fig.add_subplot(gs[2, 1])  # Twitter top k% (bottom-right)


############################################
####### REGULAR SCATTER PLOT (bsky) #######
############################################


bsky_scatter = sns.scatterplot(
    data=raw_bs_strength_change,
    x="strength_naive",
    y="strength_reconstruct",
    color="none",
    alpha=0.7,
    edgecolor=bsky_color,
    linewidth=1,
    size="count_clipped",
    sizes=(10, 500),
    ax=ax1,
    legend=False,
)

max_value = max(
    raw_bs_strength_change["strength_naive"].max(),
    raw_bs_strength_change["strength_reconstruct"].max(),
)
ax1.plot(
    [-1] + list(range(int(max_value))),
    [-1] + list(range(int(max_value))),
    color="k",
    linestyle="dashed",
    alpha=0.5,
    zorder=0,
)

ax1.set_yscale("symlog")
ax1.set_xscale("symlog")

tick_positions = [-1, 0, 1, 10, 100]
ax1.set_yticks(tick_positions, tick_positions)
ax1.set_xticks(tick_positions, tick_positions, rotation=45)

ax1.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax1.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

ax1.set_ylim(-1, 500)
ax1.set_xlim(-1, 500)

ax1.set_ylabel("reconstructed strength")
ax1.set_xlabel("naive strength")


twitter_scatter = sns.scatterplot(
    data=raw_mid_strength_change,
    x="strength_naive",
    y="strength_reconstruct",
    color="none",
    alpha=0.7,
    edgecolor=twitter_color,
    linewidth=1,
    size="count_clipped",
    sizes=(10, 500),
    ax=ax4,
    legend=False,
)

max_value = max(
    raw_mid_strength_change["strength_naive"].max(),
    raw_mid_strength_change["strength_reconstruct"].max(),
)
ax4.plot(
    [-1] + list(range(int(max_value))),
    [-1] + list(range(int(max_value))),
    color="k",
    linestyle="dashed",
    alpha=0.5,
    zorder=0,
)

ax4.set_yscale("symlog")
ax4.set_xscale("symlog")

tick_positions = [-1, 0, 1, 10, 100, 1000, 10000]
ax4.set_yticks(tick_positions, tick_positions)
ax4.set_xticks(tick_positions, tick_positions, rotation=45)

ax4.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax4.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

ax4.set_ylim(-1, 10000)
ax4.set_xlim(-1, 10000)

ax4.set_ylabel("reconstructed strength")
ax4.set_xlabel("naive strength")

#########################
###### STRIP PLOT ######
#########################

sns.stripplot(
    data=bsky_df,
    x="k",
    y="jaccard_sim",
    orient="verticle",
    s=4,
    color=bsky_color,
    marker="x",
    alpha=0.75,
    jitter=0.25,
    ax=ax3,
    linewidth=1,
)


sns.stripplot(
    data=mid_df,
    x="k",
    y="jaccard_sim",
    orient="verticle",
    s=5,
    color=twitter_color,
    marker="x",
    alpha=0.75,
    jitter=0.25,
    linewidth=1,
    ax=ax6,
)

ax3.set_ylabel("similarity\n(reconstructed vs. naive)")
ax6.set_ylabel("similarity\n(reconstructed vs. naive)")

ax3.set_xlabel(r"top $k$% influentials")
ax6.set_xlabel(r"top $k$% influentials")

# Rotate x-axis labels
ax3.tick_params(axis="x", rotation=45)
ax6.tick_params(axis="x", rotation=45)

ax3.grid(axis="y")
ax6.grid(axis="y")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)


##################################
###### SCATTER PLOT BLUESKY ######
##################################


bskyfig = sns.scatterplot(
    data=bs_strength_change_count,
    x="strength_naive",
    y="mean_strength_diff_recon_minus_naive",
    size="count_clipped",
    sizes=(10, 500),
    alpha=0.5,
    color="none",
    edgecolor=bsky_color,
    ax=ax2,
    linewidth=1,
    legend=False,
    zorder=3,
)

# Median change
ax2.scatter(
    bs_median_strength_diff["strength_naive"],
    bs_median_strength_diff["median_strength_diff"],
    color="red",
    marker="x",
    zorder=5,
    s=10,
)


ax2.hlines(
    y=0,
    xmin=0,
    xmax=bs_strength_change["strength_naive"].max(),
    linestyle="dashed",
    color="k",
    alpha=0.5,
    zorder=0,
)

ax2.set_xscale("symlog")
ax2.set_yscale("symlog")

ax2.set_xlim(-1, 500)

tick_positions = [-1, 0, 1, 10, 100]
ax2.set_xticks(tick_positions, tick_positions, rotation=45)

ax2.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax2.xaxis.set_major_formatter(FuncFormatter(custom_formatter))


ax2.set_ylabel("avg. strength change")
ax2.set_xlabel("naive strength")


##################################
###### SCATTER PLOT TWITTER ######
##################################


twitfig = sns.scatterplot(
    data=mid_strength_change_count,
    x="strength_naive",
    y="mean_strength_diff_recon_minus_naive",
    size="count_clipped",
    sizes=(10, 500),
    alpha=0.5,
    color="none",
    edgecolor=twitter_color,
    ax=ax5,
    linewidth=1,
    zorder=3,
)

ax5.scatter(
    mid_median_strength_diff["strength_naive"],
    mid_median_strength_diff["median_strength_diff"],
    color="red",
    marker="x",
    zorder=5,
    s=10,
)


ax5.hlines(
    y=0,
    xmin=0,
    xmax=mid_strength_change["strength_naive"].max(),
    linestyle="dashed",
    color="k",
    alpha=0.5,
    zorder=0,
)


ax5.set_xscale("symlog")
ax5.set_yscale("symlog")

tick_positions = [-1, 0, 1, 10, 100, 1000, 10000]
ax5.set_xticks(tick_positions, tick_positions, rotation=45)

ax5.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax5.xaxis.set_major_formatter(FuncFormatter(custom_formatter))


ax5.set_ylabel(
    "avg. strength change",
)
ax5.set_xlabel("naive strength")


ax1.annotate("(a)", xy=(-0.3, 1.1), xycoords=ax1.transAxes)
ax2.annotate("(b)", xy=(-0.3, 1.1), xycoords=ax2.transAxes)
ax3.annotate("(c)", xy=(-0.3, 1.1), xycoords=ax3.transAxes)
ax4.annotate("(d)", xy=(-0.3, 1.1), xycoords=ax4.transAxes)
ax5.annotate("(e)", xy=(-0.3, 1.1), xycoords=ax5.transAxes)
ax6.annotate("(f)", xy=(-0.3, 1.1), xycoords=ax6.transAxes)


# Create a temporary scatter plot with legend to extract handles
temp_scatter = sns.scatterplot(
    data=raw_mid_strength_change,
    x="strength_naive",
    y="strength_reconstruct",
    size="count_clipped",
    sizes=(10, 500),
    alpha=0,
    ax=ax5,
    legend="full",
)

# Get legend handles from the temporary scatter plot
sizes_legend = [100, 300, 500]
handles, labels = temp_scatter.get_legend_handles_labels()
selected_handles = []
selected_labels = []

# Remove the temporary legend
if ax5.get_legend() is not None:
    ax5.get_legend().remove()

all_sizes = [float(val) for val in labels]

# Only keep unique values that match our desired sizes
seen_sizes = set()
for size, handle, label in zip(all_sizes, handles, labels):
    for target_size in sizes_legend:
        if np.isclose(float(label), target_size) and target_size not in seen_sizes:
            handle.set_alpha(1)
            selected_handles.append(handle)
            selected_labels.append(label)
            seen_sizes.add(target_size)
            break

# Add the legend centered at the top of the figure, above subplots (a) and (d)
if selected_labels:
    selected_labels[-1] = r"$\geq$" + f"{selected_labels[-1]}"
    legend = fig.legend(
        selected_handles,
        selected_labels,
        ncols=3,
        title="number of nodes",
        loc="center",
        fontsize=10,
        title_fontsize=10,
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )

    # Adjust the title's vertical alignment manually using padding
    legend.get_title().set_position((0, 5))

plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Define the file name and extensions
file_name = "top_k_influence_comparison"
extensions = ["pdf", "png", "svg"]

# Save the figure in different formats using a loop
print(f"Saving figures to {OUTPUT_DIR}/")
for ext in extensions:
    output_path = os.path.join(OUTPUT_DIR, f"{file_name}.{ext}")
    fig.savefig(output_path, dpi=800, bbox_inches="tight")
    print(f"\t- {output_path} created")
