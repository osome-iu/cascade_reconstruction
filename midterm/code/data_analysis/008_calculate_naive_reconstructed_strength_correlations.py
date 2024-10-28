"""
Purpose:
    Calculate the correlation between the node strengths in the reconstructed
    network versus the naive network. 

Inputs:
    None, data loaded with constants below

Output:
    - .parquet dataframe with the following columns:
        - "net_ver": the network version number for PDI reconstruction
        - "gamma": gamma value for PDI reconstruction
        - "alpha": alpha value for PDI reconstruction
        - "spearman_r": the Spearman correlation coefficient
        - "pvalue": the p-value for the Spearman correlation 

Authors: 
- Matthew DeVerna
"""

import os

import pandas as pd

from scipy import stats

STRENGTH_DIFFS_DIR = (
    "/data_volume/cascade_reconstruction/networks_stats/strength_differences/"
)

# Get the list of files
bs_files = os.listdir(STRENGTH_DIFFS_DIR)

# This removes the mean difference files so that we're only calculating correlations between node strengths
bs_files = sorted([file for file in bs_files if file.startswith("strength_change")])

bs_correlation_records = []

print("Beginning Midterm data analysis...")
for file in bs_files:
    print(f"\t - {file}")
    temp_df = pd.read_parquet(os.path.join(STRENGTH_DIFFS_DIR, file))

    # Iterate through the network versions.
    # NOTE: including gamma and alpha below is redundant, but allows easier extraction for each record
    for ver_gamma_alpha, temp_df in temp_df.groupby(["net_v", "gamma", "alpha"]):
        ver, gamma, alpha = ver_gamma_alpha

        # Calculate correlation and store record
        result = stats.spearmanr(
            temp_df["strength_naive"], temp_df["strength_reconstruct"]
        )
        bs_correlation_records.append(
            {
                "net_ver": ver,
                "gamma": gamma,
                "alpha": alpha,
                "spearman_r": result.statistic,
                "pvalue": result.pvalue,
            }
        )

# Create and save dataframe
bs_correlation_df = pd.DataFrame.from_records(bs_correlation_records)
fname = f"midterm_node_strength_correlations.parquet"
outpath = os.path.join(STRENGTH_DIFFS_DIR, fname)
bs_correlation_df.to_parquet(outpath, index=False, engine="pyarrow")
