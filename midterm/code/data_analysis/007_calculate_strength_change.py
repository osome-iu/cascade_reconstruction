"""
Purpose:
    Calculate the change in strength for individual nodes across all reconstructed
    midterm networks as compared with the naive network. Also, calculate the mean
    change for each node.

Inputs:
    .parquet node centrality files

Outputs:
    1. .parquet files in output dir specified by constants (see below)
        - Rows represent nodes and Columns are:
            - 'net_v': the reconstructed network version
            - 'gamma': the gamma value used in the reconstruction
            - 'alpha': the alpha value used in the reconstruction
            - 'user_id': the user ID of the node (duplicated across the above parameters)
            - 'strength_reconstruct': the strength value after reconstruction
            - 'strength_naive': the strength value in the naive network
            - 'strength_diff_recon_minus_naive': the difference between the two strength
                values (for this user_id/node), subtracting the naive value from the reconstruction.
                We do it this way because the resulting sign/direction of the difference
                value is more intuitive. For example, if it is positive it means the node's
                strength value increased.
    2. .parquet files in output dir specified by constants (see below)
        - Rows respresent nodes and Columns are:
            - 'user_id': the user ID of the node (duplicated across the above parameters)
            - 'strength_naive': the strength value in the naive network
            - 'mean_recon_minus_naive_diff': the mean difference between the two strength
                values (for this user_id/node), subtracting the naive value from the reconstruction.
                We do it this way because the resulting sign/direction of the difference
                value is more intuitive. For example, if it is positive it means the node's
                strength value increased.

Authors:
    Matthew DeVerna
"""

import os

import pandas as pd

from midterm.utils import collect_files_recursively

NAIVE_NET_FNAME = "naive_network_centralities.parquet"
CENTRALITIES_DIR = "/data_volume/cascade_reconstruction/networks_stats/centralities"
OUTPUT_DIR = "/data_volume/cascade_reconstruction/networks_stats/strength_differences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GAMMAS = [0.25, 0.5, 0.75]
ALPHAS = [1.1, 2.0, 3.0]


def generate_file_list_dict(data_dir, gammas, alphas):
    """
    Return a dictionary mapping gamma/alpha paramers to a list of file paths
    that match those parameters

    Parameters
    ----------
    - data_dir (str) : full path to a directory
    - gammas (List[float]): list of gamma values to match
    - alphas (List[float]): list of alpha values to match

    Return
    ----------
    - file_list_dict (dict) : dictionary with following form
        {
            matching_string1 : [file/path/one.ext, file/path/two.ext, ...],
            matching_string2 : [file/path/one.ext, file/path/two.ext, ...],
        }

    Note: matching_stringX will look like: "*gamma_0.25_alpha_3.0*" for each
        gamma and alpha value passed.
    """

    file_list_dict = dict()
    for gamma in gammas:
        for alpha in alphas:
            matching_string = f"*gamma_{gamma}_alpha_{alpha}*"
            file_list = sorted(collect_files_recursively(matching_string, data_dir))
            clean_param_str = matching_string.replace("*", "")
            file_list_dict[clean_param_str] = file_list

    return file_list_dict


if __name__ == "__main__":

    # Load the naive network centralities
    naive_file_path = os.path.join(CENTRALITIES_DIR, NAIVE_NET_FNAME)
    naive_df = pd.read_parquet(naive_file_path)

    file_list_dict = generate_file_list_dict(CENTRALITIES_DIR, GAMMAS, ALPHAS)
    for params, file_paths in file_list_dict.items():
        print(f"Working on : {params}")

        # Load all of the centrality values for each node across
        # all PDI network versions and combine them into one dataframe.
        pdi_dfs = []
        for ver, file_path in enumerate(file_paths, start=1):
            print(f"\t- File version: {ver}")
            pdi_dfs.append(pd.read_parquet(file_path))
        pdi_df = pd.concat(pdi_dfs)

        # Merge this with the naive network and take the difference
        merged_df = pd.merge(
            left=pdi_df[["net_v", "gamma", "alpha", "user_id", "strength"]],
            right=naive_df[["user_id", "strength"]],
            on="user_id",
            suffixes=["_reconstruct", "_naive"],
        )
        merged_df["strength_diff_recon_minus_naive"] = (
            merged_df["strength_reconstruct"] - merged_df["strength_naive"]
        )

        # Save these node-level differences (with duplicate nodes)
        output_fname = f"strength_change_{params}.parquet"
        output_path = os.path.join(OUTPUT_DIR, output_fname)
        merged_df.to_parquet(output_path)

        # Calculate the mean strength change for each user ID
        mean_strength_diff_recon_minus_naive = (
            merged_df.groupby(["user_id", "strength_naive"])[
                "strength_diff_recon_minus_naive"
            ]
            .mean()
            .to_frame("mean_strength_diff_recon_minus_naive")
            .reset_index()
        )

        # Save the mean differences
        output_fname = f"mean_strength_change_{params}.parquet"
        output_path = os.path.join(OUTPUT_DIR, output_fname)
        mean_strength_diff_recon_minus_naive.to_parquet(output_path)
