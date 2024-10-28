"""
Purpose:
    Generate statistics reported about the influence change six-panel figure in the paper.

Inputs:
    - None. Data read via constants/paths defined in the script.

Outputs:
    - .txt file with the statistics.

Author:
    Matthew DeVerna
"""

import os

import pandas as pd


# Change the current working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "influence_change_stats.txt")

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
    & (bsky_df.k < 15)
]

mid_df = pd.read_parquet(mid_jaccard_file)
mid_df = mid_df[
    (mid_df.metric == "strength")
    & (mid_df.gamma == 0.25)
    & (mid_df.alpha == 3.0)
    & (mid_df.k < 15)
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

### Calculate statistics reported in the manuscript
# Calculate the proportion of positive strength differences
bs_positive_proportion = sum(
    bs_strength_change["mean_strength_diff_recon_minus_naive"] > 0
) / len(bs_strength_change)
mid_positive_proportion = sum(
    mid_strength_change["mean_strength_diff_recon_minus_naive"] > 0
) / len(mid_strength_change)

# Calculate the mean Jaccard similarity for each platform at each k value
bsky_mean_jaccard = (
    bsky_df.groupby("k")["jaccard_sim"].mean().to_frame("mean_jaccard").reset_index()
)
mid_mean_jaccard = (
    mid_df.groupby("k")["jaccard_sim"].mean().to_frame("mean_jaccard").reset_index()
)


# Write statistics to the output file
with open(OUTPUT_FILE, "w") as f:
    f.write(f"{'Influence Change Statistics'.upper()}\n")

    f.write("Proportion of positive strength differences:\n")
    f.write(f"Bluesky: {bs_positive_proportion:.2%}\n")
    f.write(f"Twitter: {mid_positive_proportion:.2%}\n\n")

    f.write("Mean Jaccard similarity for Bluesky at each k value:\n")
    f.write(bsky_mean_jaccard.to_string(index=False))
    f.write("\n\n")

    f.write("Mean Jaccard similarity for Twitter at each k value:\n")
    f.write(mid_mean_jaccard.to_string(index=False))
    f.write("\n")
