"""
Purpose:
    Calculate cascade similarity metrics. See columns below for details.

Note:
    - We return mean values only as saving all 4,950 comparisons for each of the
        cascades would create > 200M rows for each alpha level:
        --> 4,950 comparisons * ~40,000 cascades = 198,000,000

Output:
    - Pandas dataframe of cascade similarity metrics.
    - Columns:
        - alpha (float): the alpha value
        - cascade_id (str): the ID of the cascade
        - size (int): size of the cascade (number of nodes)
        - jaccard_mean (float): the mean jaccard similarity between the set of edges in g1
            vs. g2 where g1 and g2 are different versions of the same cascade.
            --> i.e., intersection_edges_size / edges_union_size
        - prop_mismatched_parents_mean (float): the mean proportion of edges between the set
            of edges in g1 vs. g2 where g1 and g2 are different versions of the same cascade.
            --> i.e., disjoint_edges_size / num_edges
            --> Note that num_edges is necessarily the same for both cascades
        - {metric}_ci_low (float): the lower bound of a bootstrapped 95% confidence interval
        - {metric}_ci_hig (float): the upper bound of a bootstrapped 95% confidence interval
            --> "metric" is replaced by both 'jaccard' or 'prop_mismatched_parents' for each
                metric above

Author:
- Matthew DeVerna
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pickle as pkl

from igraph import Graph
from itertools import combinations
from pathlib import Path

from pkg.boot import bootstrap_ci

REC_DATA_DIR = "../../output/reconstructed_data"
OUTPUT_DIR = "../../output/cascade_similarity_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_DIRS = ["alpha_1_1", "alpha_1_5", "alpha_2_0", "alpha_2_5", "alpha_3_0"]
GAMMA_DIRS = ["gamma_0_25", "gamma_0_5", "gamma_0_75"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate mean cascade similarity metrics."
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["midterm", "vosoughi"],
        help="Specify the cascade type to analyze.",
    )
    parser.add_argument(
        "--gamma",
        required=True,
        choices=GAMMA_DIRS,
        help=f"Select the desired alpha for reconstructed cascades.",
    )
    parser.add_argument(
        "--alpha",
        required=True,
        choices=ALPHA_DIRS,
        help="Select the desired alpha for reconstructed cascades.",
    )
    parser.add_argument(
        "--tid",
        action="store_true",
        help="Include to calculate similarity metrics in comparison to the TID cascades. Exclude to compare all PDI cascades to themselves.",
    )
    args = parser.parse_args()
    return args


def get_jaccard_mismatched(g1_edge_set, g2_edge_set):
    """
    Calculate Jaccard similarity and proportion of mismatched parents between two sets of edges.

    Parameters:
    - g1_edge_set (set): A set of edges representing the first graph.
    - g2_edge_set (set): A set of edges representing the second graph.

    Returns:
    - jaccard_similarity (float): The Jaccard similarity between the two edge sets.
    - prop_mismatched_parents (float): The proportion of mismatched parents between the two edge sets.
    """
    # Denominators for later
    edges_union_size = len(g1_edge_set.union(g2_edge_set))
    num_edges = len(g1_edge_set)  # Equal for both cascade versions

    # Levenshtein distance, since node set is necessarily equal
    disjoint_edges_size = len(g1_edge_set.difference(g2_edge_set))

    intersection_edges_size = len(g1_edge_set.intersection(g2_edge_set))

    # Calculate and save for each comparison
    jaccard_similarity = intersection_edges_size / edges_union_size
    prop_mismatched_parents = disjoint_edges_size / num_edges

    return jaccard_similarity, prop_mismatched_parents


if __name__ == "__main__":
    args = parse_args()
    tid = args.tid
    gamma_dir = args.gamma
    alpha_dir = args.alpha
    cascade_type = f"pdi_{args.type}"
    if cascade_type == "pdi_midterm":
        raise Exception("Not implemented yet! Must update for new midterm data!")
    cascade_dir_base = os.path.join(REC_DATA_DIR, cascade_type)

    if tid:
        tid_files = sorted(
            glob.glob(os.path.join(REC_DATA_DIR, "time_inferred_diffusion/*pkl"))
        )
        tid_graphs = {}
        for file in tid_files:
            with open(file, "rb") as f:
                graph_list = pkl.load(f)
                for graph in graph_list:
                    tid_graphs[graph.cascade_id] = graph

    alpha_path = os.path.join(cascade_dir_base, gamma_dir, alpha_dir)
    cascade_ids = os.listdir(alpha_path)
    num_cascades = len(cascade_ids)

    # Handles symbolic link weirdness, if present.
    alpha_path = Path(alpha_path).resolve()

    gamma_val = float(".".join(gamma_dir.split("_")[1:]))
    alpha_val = float(".".join(alpha_dir.split("_")[1:]))
    print(f"Working on (gamma, alpha): ({gamma_val}, {alpha_val})...")

    records = []
    for cas_num, cascade_id in enumerate(cascade_ids, start=1):
        print(f"\t- {cascade_id} ({cas_num} / {num_cascades})")

        # Get all cascade version files
        cas_dir = os.path.join(alpha_path, cascade_id)
        cascade_version_files = sorted(os.listdir(cas_dir))

        # Skip cascades with only one version (length == 2)
        if len(cascade_version_files) == 1:
            continue

        # Load all versions of a cascade
        edge_sets = []
        for cas_version in cascade_version_files:
            cas_version_path = os.path.join(cas_dir, cas_version)
            graph = Graph.Read_GraphMLz(cas_version_path)
            # Create edge set using 'names' (tids)
            # Using 'names' is more explicit as the vertex indices (which count up from 0)
            # many not match across cascade versions
            g_edge_set = set(
                (e.source_vertex["name"], e.target_vertex["name"]) for e in graph.es
            )
            edge_sets.append(g_edge_set)

        if tid:
            # Compare each graph version to the TID cascade
            tid_graph = tid_graphs[cascade_id]
            # Create edge set using 'names' (tids)
            tid_edge_set = set(
                (e.source_vertex["name"], e.target_vertex["name"]) for e in tid_graph.es
            )
            edge_sets_comparisons = [(tid_edge_set, e) for e in edge_sets]
        else:
            # Compare all versions to one another
            edge_sets_comparisons = list(combinations(edge_sets, 2))

        jaccard_list = []
        prop_mismatched_list = []
        for e1, e2 in edge_sets_comparisons:
            jaccard, prop_mismatched = get_jaccard_mismatched(e1, e2)
            jaccard_list.append(jaccard)
            prop_mismatched_list.append(prop_mismatched)

        # Bootstrap 95% confidence intervals (default is 1000 resamples)
        jaccard_ci_low, jaccard_ci_high = bootstrap_ci(jaccard_list)
        prop_mismatch_ci_low, prop_mismatch_ci_high = bootstrap_ci(prop_mismatched_list)
        records.append(
            {
                "gamma": gamma_val,
                "alpha": alpha_val,
                "cascade_id": cascade_id,
                "size": graph.vcount(),
                "jaccard_mean": np.mean(jaccard_list),
                "prop_mismatched_parents_mean": np.mean(prop_mismatched_list),
                "jaccard_ci_low": jaccard_ci_low,
                "jaccard_ci_hi": jaccard_ci_high,
                "prop_mismatched_parents_ci_low": prop_mismatch_ci_low,
                "prop_mismatched_parents_ci_high": prop_mismatch_ci_high,
            }
        )

    print(f"Saving cascade stats for...")
    print(f"\t- gamma = {gamma_val}, alpha = {alpha_val}...")
    # Save results as parquet (we do not lose intermediate results!)
    similarity_df = pd.DataFrame.from_records(records)
    fname_ending = "vs_tid_version" if tid else "vs_pdi_versions"
    fname = f"cascade_similarity_metrics_{fname_ending}_{gamma_dir}_{alpha_dir}.parquet"
    similarity_df.to_parquet(os.path.join(OUTPUT_DIR, fname))
    print("--- Script complete ---")
