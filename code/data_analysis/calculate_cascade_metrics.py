"""
Purpose:
    Calculate cascade metrics. See columns below for details.

Note:
    - Each metric will be mapped to it's cascade ID and version number.

Output:
    - Pandas dataframe of cascade metrics. Columns:
        - cascade_id (str): the ID of the cascade
        - version_num (int): the version number of the output file
        - depth (float): the maximum shortest path from the root node to any
            other node in the cascade
        - size (int): the number of nodes in the cascade
        - structural_virality (float): the average (undirecteds) shortest path in the cascade
        - max_breadth (int): the maximum breadth of the cascade. I.e. the maximum
            number of nodes at any depth/distance from the root.

Author: 
- Matthew DeVerna
"""

import argparse
import os

import pandas as pd

from collections import Counter
from igraph import Graph
from pathlib import Path

# Paths
DATA_DIR = "../../cleaned_data"
OUTPUT_DIR = "../../output/cascade_metrics"
CAS_DIR_BASE = "/data_volume/cascade_reconstruction"
ALPHA_DIRS = ["alpha_1_1", "alpha_1_5", "alpha_2_0", "alpha_2_5", "alpha_3_0"]
GAMMA_DIRS = ["gamma_0_25", "gamma_0_5", "gamma_0_75"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate cascade metrics max breadth, structural virality, size, and depth."
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["midterm", "vosoughi"],
        help="Specify the cascade type to analyze.",
    )
    exclude_note = "If excluded, analyze all possible options."
    parser.add_argument(
        "--gamma",
        choices=GAMMA_DIRS,
        help=f"Specify the gamma levels by directory name to analyze. {exclude_note}",
    )
    parser.add_argument(
        "--alpha",
        choices=ALPHA_DIRS,
        help=f"Specify the alpha levels by directory name to analyze. {exclude_note}",
    )
    args = parser.parse_args()
    return args


def calculate_max_breadth(graph, root_id):
    """
    Calculate the maximum breadth of the cascade.

    Notes: Conducts a BFS from the root node, recording the depth of each node.
        At the end, count the number of nodes at each depth, the max count will
        be the maximum breadth.

    Parameters:
    ----------
    - graph (igraph.Graph): the cascade graph
    - root_id (int): the ID of the root node. Not the "tid", which is the "name" parameter.

    Returns:
    ----------
    - max_breadth (int): the maximum breadth
    """
    depth_counter = Counter()
    # advanced=True includes the depth as we go
    for step_info in graph.bfsiter(root_id, advanced=True):
        curr_vertex, depth, parent_vertex = step_info[0], step_info[1], step_info[2]
        depth_counter[depth] += 1

    # Returns a descending sorted list of (depth, count) tuples
    return depth_counter.most_common()[0][1]


def create_cascade_to_seconds_map(df):
    """
    Generate a mapping of cascade IDs to the total number of seconds between the
    earliest and oldest posts in each cascade.
    Parameters
    ----------
    df (pandas.DataFrame): The DataFrame containing the data.
    Returns
    -------
    cascade_id_to_seconds (dict): A {cascade_id: total_seconds} dictionary
    """
    cascade_id_to_seconds = {}
    for cascade, temp_frame in df.groupby("cascade_id"):
        earliest_post = temp_frame.tweet_date.min()
        oldest_post = temp_frame.tweet_date.max()
        total_seconds = (oldest_post - earliest_post).total_seconds()
        cascade_id_to_seconds[cascade] = total_seconds
    return cascade_id_to_seconds


if __name__ == "__main__":
    # Set up paths
    args = parse_args()
    gamma_dirs = [args.gamma] if args.gamma else GAMMA_DIRS
    alpha_dirs = [args.alpha] if args.alpha else ALPHA_DIRS
    cascade_type = f"pdi_{args.type}"
    if cascade_type == "pdi_midterm":
        raise Exception("Not implemented here! handled in a different script!")
    cascade_dir_base = os.path.join(CAS_DIR_BASE, cascade_type)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Load data
    print("Creating a map from cascade ID to root tweet ID for all cascade IDs...")
    df = pd.read_parquet(os.path.join(DATA_DIR, "clean_raw_data_anon.parquet"))
    roots = df[df.parent_tid == "-1"]
    cas_to_root_tid = {}
    for cas_id, cas_root_tid in zip(roots.cascade_id, roots.cascade_root_tid):
        cas_to_root_tid[cas_id] = cas_root_tid
    print(f"\t- Done.")

    print("Creating a map from cascade ID to total seconds...")
    cascde_id_to_seconds = create_cascade_to_seconds_map(df)
    print(f"\t- Done.")

    print("Creating a map from cascade ID to number of unique users...")
    cascde_id_to_num_unique_users = df.groupby("cascade_id")["tid"].nunique().to_dict()
    print(f"\t- Done.")

    del df

    not_undirected_err_msg = f"Error, Graph was not changed to undirected!"

    for gamma_dir in gamma_dirs:
        gamma_value = ".".join(gamma_dir.split("_")[1:])

        print("=" * 50)
        print(f"Begin gamma = {gamma_value}")
        print("=" * 50)
        print("=" * 50)

        full_gamma_dir = os.path.join(cascade_dir_base, gamma_dir)

        for alpha_dir in alpha_dirs:
            alpha_value = ".".join(alpha_dir.split("_")[1:])

            print("#" * 50)
            print(f"Begin alpha = {alpha_value}")
            print("#" * 50)
            print("#" * 50)

            full_alpha_dir = os.path.join(full_gamma_dir, alpha_dir)

            # Handles symbolic link weirdness, if present.
            full_alpha_dir = Path(full_alpha_dir).resolve()

            all_cascade_ids = [
                d
                for d in os.listdir(full_alpha_dir)
                if os.path.isdir(os.path.join(full_alpha_dir, d))
            ]

            metric_records = []

            for cascade_id in all_cascade_ids:
                print(f"Generating statistics for cascade ID: {cascade_id}...")
                num_versions = len(os.listdir(os.path.join(full_alpha_dir, cascade_id)))
                for version_num in range(1, num_versions + 1):
                    cascade_path = os.path.join(
                        full_alpha_dir,
                        cascade_id,
                        f"v_{str(version_num).zfill(3)}.gmlz",
                    )
                    g = Graph.Read_GraphMLz(cascade_path)
                    g.to_undirected()  # Changes graph in place
                    assert not g.is_directed(), not_undirected_err_msg

                    cascade_root_tid = cas_to_root_tid[cascade_id]
                    root_index = g.vs.find(name=cascade_root_tid).index

                    depth = g.eccentricity(root_index)
                    structural_virality = g.average_path_length()
                    size = g.vcount()
                    max_breadth = calculate_max_breadth(g, root_id=root_index)

                    metric_records.append(
                        {
                            "cascade_id": cascade_id,
                            "version": version_num,
                            "gamma": float(gamma_value),
                            "alpha": float(alpha_value),
                            "depth": depth,
                            "structural_virality": structural_virality,
                            "size": size,
                            "max_breadth": max_breadth,
                            "total_seconds": cascde_id_to_seconds[cascade_id],
                            "num_unique_users": cascde_id_to_num_unique_users[
                                cascade_id
                            ],
                        }
                    )

            print(f"Saving cascade metric statistics for...")
            print(f"\t- gamma = {gamma_value}, alpha = {alpha_value}...")
            metric_stats_df = pd.DataFrame.from_records(metric_records)
            stats_outpath = os.path.join(
                OUTPUT_DIR,
                f"cascade_metrics_statistics_{gamma_dir}_{alpha_dir}.parquet",
            )
            metric_stats_df.to_parquet(stats_outpath)
            del metric_stats_df  # This will b > 4 million rows so del for memory.

    print("#" * 50)
    print("Script complete.")
