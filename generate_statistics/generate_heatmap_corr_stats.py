"""
Purpose:
    Generate heatmap correlation statistics.
    Correlations are between the strength of a node in a naive network and the
    strength of the same node in the reconstructed version of that same network.

Inputs:
    - None. Data read via constants/paths defined in the script.

Outputs:
    - .txt file with the correlation statistics.

Author:
    Matthew DeVerna
"""

import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTDIR = "statistics"
OUTFILE = "naive_vs_reconstructed_node_str_corrs.txt"
os.makedirs(OUTDIR, exist_ok=True)

# Each row in these files contain correlations between the strength of a node in a naive
# network and the strength of the same node in the reconstructed version of that same network..
BLUESKY_NODE_STR_CORRS = "/data_volume/cascade_reconstruction/bluesky/networks_stats/strength_differences/bluesky_node_strength_correlations.parquet"
MIDTERM_NODE_STR_CORRS = "/data_volume/cascade_reconstruction/networks_stats/strength_differences/midterm_node_strength_correlations.parquet"
bs_corrs = pd.read_parquet(BLUESKY_NODE_STR_CORRS)
mid_corrs = pd.read_parquet(MIDTERM_NODE_STR_CORRS)

# Calculate the mean correlation across all networks within each parameter setting.
bs_mean_corrs = (
    bs_corrs.groupby(["gamma", "alpha"])["spearman_r"]
    .mean()
    .to_frame("mean_correlation")
    .reset_index()
)
bs_std_corrs = (
    bs_corrs.groupby(["gamma", "alpha"])["spearman_r"]
    .std()
    .to_frame("std_correlation")
    .reset_index()
)

# Calculate the standard deviation of the correlation across all networks within each parameter setting.
mid_mean_corrs = (
    mid_corrs.groupby(["gamma", "alpha"])["spearman_r"]
    .mean()
    .to_frame("mean_correlation")
    .reset_index()
)
mid_std_corrs = (
    mid_corrs.groupby(["gamma", "alpha"])["spearman_r"]
    .std()
    .to_frame("std_correlation")
    .reset_index()
)

# Add platform column to each dataframe
bs_mean_corrs["platform"] = "bluesky"
bs_std_corrs["platform"] = "bluesky"
mid_mean_corrs["platform"] = "twitter"
mid_std_corrs["platform"] = "twitter"

# Merge mean and std dataframes for each platform
bs_corrs_combined = pd.merge(
    bs_mean_corrs, bs_std_corrs, on=["gamma", "alpha", "platform"]
)
mid_corrs_combined = pd.merge(
    mid_mean_corrs, mid_std_corrs, on=["gamma", "alpha", "platform"]
)

# Combine both platforms into one dataframe
combined_corrs = pd.concat([bs_corrs_combined, mid_corrs_combined], ignore_index=True)

# Rename columns for clarity
combined_corrs = combined_corrs.rename(
    columns={"mean_correlation_x": "mean", "std_correlation_y": "std"}
)

# Sort to match the manuscript
combined_corrs = combined_corrs.sort_values(["gamma", "alpha"])

# Open the file in write mode
output_path = os.path.join(OUTDIR, OUTFILE)
with open(output_path, "w") as f:
    f.write("NODE STRENGTH CORRELATION STATISTICS\n\n")
    for platform in ["twitter", "bluesky"]:
        platform_corrs = combined_corrs[combined_corrs["platform"] == platform]
        f.write(f"Correlation Statistics for {platform.capitalize()}:\n")
        f.write(platform_corrs.round(4).to_string(index=False))
        f.write("\n\n")
