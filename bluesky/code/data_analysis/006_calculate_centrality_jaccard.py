"""
Purpose:
    Calculate the jaccard similarity between the top-k percentile nodes in the
    naive and PDI networks based on different centrality metrics.

Inputs:
    None, networks loaded with constants below

Output:
    - .parquet dataframe with the following columns:
        - "net_version": the network version number for PDI reconstruction
        - "gamma": gamma value for PDI reconstruction
        - "alpha": alpha value for PDI reconstruction
        - "metric": the metric utilized to select top k-users
            Options: ("degree", "strength", "kcore", "eigenval")
        - "k": top percentile
            - 1 = top 1%
        - "jaccard_sim": jaccard_similarity between the PDI and naive network
            top-k users for the metric specified in the ro

Authors: 
- Matthew DeVerna
"""

import os

import numpy as np
import pandas as pd

from midterm.utils import collect_files_recursively


OUTPUT_DIR = "/data_volume/cascade_reconstruction/bluesky/networks_stats"
CENTRALITIES_DIR = os.path.join(OUTPUT_DIR, "centralities/")
NAIVE_NET_FILE = os.path.join(CENTRALITIES_DIR, "naive_network_centralities.parquet")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MATCHING_STR = "network_v*.parquet"

K_VALUES = np.array([1, 5, 10, 15, 20, 25])
METRIC_COLUMNS = ["degree", "strength", "kcore", "eigenval"]


def extract_params(file_path):
    """
    Extract network version number, gamma, and alpha parameters from full path.

    Example path:
    - ".../bluesky/networks_stats/centralities/network_v_100_gamma_0.5_alpha_1.1.parquet"

    Parameters
    ----------
    - file_path (str) : full path to network file

    Return
    ----------
    - params (dict) : dictionary with following form
        {
            "ver_num : int
            "alpha" : float,
            "gamma" : float
        }
    """
    # Extract the basename of the file without the extension
    basename, _ = os.path.splitext(os.path.basename(file_path))
    ver_num = int(basename.split("network_v_")[-1].split("_")[0])
    gamma = float(basename.split("gamma_")[-1].split("_")[0])
    alpha = float(basename.split("alpha_")[-1].split("_")[0])
    return {
        "ver_num": ver_num,
        "gamma": gamma,
        "alpha": alpha,
    }


def select_top_k(df, col_name, k):
    """
    Given a dataframe, select the top k nodes based on a specified column.

    Parameters
    ----------
    - df (pandas.DataFrame: the data to select from
    - col_name (str): the column on which to base percentile calculations
    - k (np.array): the k value on which to select

    Returns
    ----------
    selected_rows (dict): dataframe where the columns are ['user_id', `col_name`]
        and rows are the selected nodes based on the associated k_value
    """
    threshold = np.percentile(df[col_name], 100 - k)
    return df[df[col_name] >= threshold][["user_id", col_name]].reset_index(drop=True)


def calc_jaccard_similarity(set1, set2):
    """
    Calculate the jaccard similarity of two sets.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


if __name__ == "__main__":

    print("Loading naive centralities and selecting top k users for each metric...")
    naive_centralities = pd.read_parquet(NAIVE_NET_FILE)

    # {(metric col name, k) : topk_df_rows}
    top_naive = {
        (metric, k): select_top_k(naive_centralities, metric, k)
        for metric in METRIC_COLUMNS
        for k in K_VALUES
    }

    # Select all PDI files
    centrality_files = collect_files_recursively(
        matching_str=MATCHING_STR, dirname=CENTRALITIES_DIR
    )

    print("Begin iterating through files and calculating comparisons...")
    jaccard_records = []
    for file_path in sorted(centrality_files):
        print(f"\t- Working on: {os.path.basename(file_path)}")

        # Load file and parameters
        pdi_centralities = pd.read_parquet(file_path)
        pdi_params = extract_params(file_path)

        for metric in METRIC_COLUMNS:
            for k in K_VALUES:
                naive_set = set(top_naive[(metric, k)]["user_id"])
                pdi_set = set(select_top_k(pdi_centralities, metric, k)["user_id"])
                jaccard_similarity = calc_jaccard_similarity(naive_set, pdi_set)

                jaccard_records.append(
                    {
                        "net_version": pdi_params["ver_num"],
                        "gamma": pdi_params["gamma"],
                        "alpha": pdi_params["alpha"],
                        "metric": metric,
                        "k": k,
                        "jaccard_sim": jaccard_similarity,
                        "bigger_set_size": max(len(naive_set), len(pdi_set)),
                    }
                )

    jaccard_df = pd.DataFrame(jaccard_records)
    jaccard_df.to_parquet(
        os.path.join(OUTPUT_DIR, "jaccard_coefficients_pdi_vs_naive.parquet"),
        index=False,
    )

    print("--- Script complete ---")
