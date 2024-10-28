"""
Purpose:
    Calculate missing metrics for the cascades reconstructed with the time inferred diffusion method.

Note:
    Below we skip cascade_id = "0". This represents a weird edge case where the (made up) tweet ID
    is too long for igraph to store it as a 'name' variable. It's default behavior is to cut it short.
    That breaks everything that follows. The size of this cascade = 1, so we simply skip it. The max
    'name' size allowed seems to be 10. tweet_id lengths in the entire dataset are as seen below.
    ```
    dfs = []
    for file in glob.glob("../cleaned_data/v_0*.parquet"):
        dfs.append(pd.read_parquet(file))
    df = pd.concat(dfs)
    Counter(list(df['tid'].map(len)) + list(df['cascade_root_tid'].map(len)))
    -----------------------------------------------------
    Counter({10: 1, <-- cascade_id = 0 'tid'
            1: 20,
            6: 1793645,
            7: 6027852,
            4: 38927,
            5: 159666,
            3: 1788,
            2: 180,
            18: 1}) <-- cascade_id = 0 'cascade_root_tid'
    ```

Input:
    None. Paths are set and data is loaded in.

Output:
    - .parquet file: ./output/cascade_metrics/time_inferred_diffusion_metrics.parquet
    - Columns: ['cascade_id', 'depth', 'structural_virality', 'max_breadth']
    - Definitions of the above are the same as in calculate_cascade_metrics.py

Authors:
- Matthew DeVerna
"""

import argparse
import glob
import os

import pandas as pd
import pickle as pkl

from collections import Counter

DATA_DIR = "../../cleaned_data"
TID_CASCADES_DIR = "../../output/reconstructed_data/time_inferred_diffusion"
TID_ALL_CASCADES_DIR = (
    "../../output/reconstructed_data/time_inferred_diffusion_all_cascades"
)
METRICS_DIR = "../../output/cascade_metrics"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate metrics for cascades based on the time inferred diffusion method"
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Include to calculate metrics for all cascades. Exclude for only cascades with length >= 2 and without 'non' values",
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


if __name__ == "__main__":
    args = parse_args()
    calculate_all = args.all
    cascades_dir = TID_ALL_CASCADES_DIR if calculate_all else TID_CASCADES_DIR

    print("Loading data...")
    if calculate_all:
        print("All cascades selected.")
        dfs = []
        for file in glob.glob(os.path.join(DATA_DIR, "v_0*.parquet")):
            dfs.append(pd.read_parquet(file))
        df = pd.concat(dfs)
    else:
        print("Subset of clean cascades selected.")
        df = pd.read_parquet(os.path.join(DATA_DIR, "clean_raw_data_anon.parquet"))
    print(f"\t- Success.")

    print("Creating a map from cascade ID to root tweet ID for all cascade IDs...")
    roots = df[df.parent_tid == "-1"]
    cas_to_root_tid = {}
    for cas_id, cas_root_tid in zip(roots.cascade_id, roots.cascade_root_tid):
        cas_to_root_tid[cas_id] = cas_root_tid
    print(f"\t- Done.")

    print("Loading graph files...")
    graph_files = sorted(
        glob.glob(os.path.join(cascades_dir, "time_inferred_diffusion_cascades_*.pkl"))
    )
    graph_list = []
    for file in graph_files:
        print(f"\t- Loading {os.path.basename(file)}...")
        with open(file, "rb") as pkl_file:
            graph_list.extend(pkl.load(pkl_file))
    print(f"\t- Done.")

    print(f"Calculating metrics for {len(graph_list)} graphs...")
    metric_records = []
    num_casades = len(graph_list)
    for cas_num, g in enumerate(graph_list):
        g.to_undirected()  # Changes graph in place
        assert not g.is_directed()

        cascade_id = g.cascade_id
        if cascade_id == "0":
            continue
        print(f"\t- Working on cascade {cascade_id} ({cas_num}/{num_casades})...")

        cascade_root_tid = cas_to_root_tid[cascade_id]
        root_index = g.vs.find(name=cascade_root_tid).index

        depth = g.eccentricity(root_index)
        structural_virality = g.average_path_length()
        size = g.vcount()
        max_breadth = calculate_max_breadth(g, root_id=root_index)

        metric_records.append(
            {
                "cascade_id": cascade_id,
                "depth": depth,
                "structural_virality": structural_virality,
                "size": size,
                "max_breadth": max_breadth,
            }
        )
    print(f"\t- Done.\n")

    metric_records_df = pd.DataFrame.from_records(metric_records)
    fname = "time_inferred_diffusion_metrics.parquet"
    fname_all = "time_inferred_diffusion_metrics_all.parquet"
    fname = fname_all if calculate_all else fname
    output_full_path = os.path.join(METRICS_DIR, fname)
    print(f"\t- Saving {output_full_path}")
    metric_records_df.to_parquet(output_full_path)
    print("Script complete.")
