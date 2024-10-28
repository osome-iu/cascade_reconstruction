"""
Purpose:
    Generate statistics and comparisons of complementary cumulative distribution functions (CCDF) for cascade metrics.

Inputs:
    - None. Data is read from predefined directories and files.

Outputs:
    - .txt file with the CCDF comparisons and significance statistics.

Author:
    Matthew DeVerna
"""

import os

import numpy as np
import pandas as pd

import itertools
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

# Ensure the current working directory is the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))


CAS_METS_DIR = "../output/cascade_metrics"
ALPHA_DIRS = ["alpha_1_1", "alpha_1_5", "alpha_2_0", "alpha_2_5", "alpha_3_0"]
GAMMA_DIRS = ["gamma_0_25", "gamma_0_5", "gamma_0_75"]
OUTPUT_DIR = "statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FN = "ccdf_comparisons.txt"


def significance_marker(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


# Load the reconstructed data cascade metric statistics
recon_data_df = []
for gamma_dir in GAMMA_DIRS:
    for alpha_dir in ALPHA_DIRS:
        path = os.path.join(
            CAS_METS_DIR,
            f"cascade_metrics_statistics_{gamma_dir}_{alpha_dir}.parquet",
        )
        recon_data_df.append(pd.read_parquet(path))
recon_data_df = pd.concat(recon_data_df)

# Load the time inferred diffusion metrics
tid_fname = os.path.join(CAS_METS_DIR, "time_inferred_diffusion_metrics.parquet")
tid_data = pd.read_parquet(tid_fname)

# Calculate the mean cascade metrics for each cascade
recon_mean_data = (
    recon_data_df.groupby(["cascade_id", "gamma", "alpha"])[
        ["depth", "structural_virality", "max_breadth", "size"]
    ]
    .mean()
    .reset_index()
)

### Generate ccdf distributions for each metric ###
gammas = [0.25, 0.5, 0.75]
alphas = [1.1, 2.0, 3.0]
distributions = dict()

# Handle the reconstructed data
for idx_g, gamma in enumerate(gammas):
    for idx_a, alpha in enumerate(alphas):

        selected_data = recon_mean_data[
            (recon_mean_data.alpha == alpha) & (recon_mean_data.gamma == gamma)
        ]

        for loc, metric in enumerate(["depth", "structural_virality", "max_breadth"]):
            x_vals = sorted(selected_data[metric])
            y_vals = 1 - (np.array(range(len(x_vals))) / len(x_vals))
            distributions[(gamma, alpha, metric)] = x_vals

# Handle the time inferred diffusion data
x_vals = sorted(tid_data["depth"])
y_vals = 1 - (np.array(range(len(x_vals))) / len(x_vals))
distributions[("tid", "tid", "depth")] = x_vals

x_vals = sorted(tid_data["structural_virality"])
y_vals = 1 - (np.array(range(len(x_vals))) / len(x_vals))
distributions[("tid", "tid", "structural_virality")] = x_vals

x_vals = sorted(tid_data["max_breadth"])
y_vals = 1 - (np.array(range(len(x_vals))) / len(x_vals))
distributions[("tid", "tid", "max_breadth")] = x_vals

### Generate the Kolmogorov-Smirnov stats ###

# List to store the results
results = []

# Get all the keys and metrics from the dictionary
keys = list(distributions.keys())
metrics = set([key[2] for key in keys])

# Iterate over each metric
for metric in metrics:
    # Filter keys by the current metric
    metric_keys = [key for key in keys if key[2] == metric]

    # Compare all pairs of distributions for this metric
    for (gamma1, alpha1, _), (gamma2, alpha2, _) in itertools.combinations(
        metric_keys, 2
    ):
        dist1 = distributions[(gamma1, alpha1, metric)]
        dist2 = distributions[(gamma2, alpha2, metric)]

        # Perform the Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(dist1, dist2)

        # Store the results
        results.append([gamma1, alpha1, gamma2, alpha2, metric, statistic, p_value])

# Convert to a distributionsFrame
results_df = pd.DataFrame(
    results,
    columns=["gamma1", "alpha1", "gamma2", "alpha2", "metric", "statistic", "p_value"],
)

# Apply Bonferroni correction within each metric
adjusted_p_values = []

for metric in metrics:
    # Filter results by metric
    metric_df = results_df[results_df["metric"] == metric]

    # Apply Bonferroni correction to p-values within this metric
    _, p_value_adj, _, _ = multipletests(metric_df["p_value"], method="bonferroni")

    # Store the adjusted p-values
    adjusted_p_values.extend(p_value_adj)

# Add the adjusted p-values to the dataframe
results_df["p_value_adj"] = adjusted_p_values

# Apply the function to create a new column for significance
results_df["sig"] = results_df["p_value_adj"].apply(significance_marker)

# Rename the metrics for the table
results_df["metric"] = results_df["metric"].map(
    {
        "structural_virality": "str. vir.",
        "depth": "depth",
        "max_breadth": "max. breadth",
    }
)


# Open the output file
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FN)
with open(output_path, "w") as f:
    # Print the percentage of significant comparisons
    sig_count = results_df["sig"].str.contains("*", regex=False).sum()
    total_count = len(results_df)
    sig_percentage = sig_count / total_count
    f.write(
        f"Significant comparisons: {sig_count}/{total_count} ({sig_percentage:.2%}%)\n\n"
    )

    # Iterate over each metric
    for met in ["depth", "max. breadth", "str. vir."]:
        # Select a sub-df for the current metric
        selected_df = results_df[results_df.metric == met]

        # Reset index for this sub-df and move those indices into a column
        selected_df = selected_df.reset_index(drop=True)
        selected_df = selected_df.reset_index(drop=False)

        # rename the index column
        selected_df = selected_df.rename(columns={"index": "#"})

        # adjust the index to start from 1
        selected_df["#"] = selected_df["#"] + 1

        # sort by gamma and alpha
        selected_df = selected_df.sort_values(
            by=["gamma1", "alpha1", "gamma2", "alpha2"]
        )

        # Write the metric name as a header
        f.write(f"Metric: {met}\n")
        f.write(selected_df.round(2).to_string(index=False))
        f.write("\n\n")  # Add some space between tables
