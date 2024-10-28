"""
Purpose:
    Generate all cascades based on the time inferred diffusion method.

Note:
    We are simply using the source target maps provided in the replication code.

Input:
    None. Paths are set and data is loaded in.

Output:
    - Cascades are output to the output/reconstructed_data/time_inferred_diffusion directory.
        - Files will take the following form: time_inferred_diffusion_cascades_{str(cas_file_num).zfill(2)}.pkl
        - They will contain a list of graphs, the cascade ID is added as an attribute to the graph and
        can be accessed via `graph.cascade_id`.

Authors:
- Matthew DeVerna
"""
import argparse
import glob
import os

import pandas as pd
import pickle as pkl

from igraph import Graph

DATA_DIR = "../../cleaned_data"
OUTPUT_DIR = "../../output/reconstructed_data/time_inferred_diffusion"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate all cascades based on the time inferred diffusion method"
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Include to construct all cascades. Exclude for only cascades with length >= 2 and without 'none' values.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    construct_all = args.all

    print("Loading data...")
    if construct_all:
        print("All cascades selected.")
        dfs = []
        for file in glob.glob(os.path.join(DATA_DIR, "v_0*.parquet")):
            dfs.append(pd.read_parquet(file))
        data = pd.concat(dfs)
        outdir = f"{OUTPUT_DIR}_all_cascades"
    else:
        print("Subset of clean cascades selected.")
        data = pd.read_parquet(os.path.join(DATA_DIR, "clean_raw_data_anon.parquet"))
        outdir = OUTPUT_DIR
    print(f"\t- Success.")

    os.makedirs(outdir, exist_ok=True)

    print("Begin generating cascade files...")
    graph_list = []  # list of graphs
    cascade_ids = data.cascade_id.unique()
    num_cascades = len(cascade_ids)

    cas_file_num = 1
    for cas_num, cascade_id in enumerate(cascade_ids, start=1):
        print(f"\t- Generating cascade {cascade_id} ({cas_num}/{num_cascades})...")
        one_cascade = data[data.cascade_id == cascade_id].copy()

        vertices = set()
        edge_list = []
        for idx, row_data in one_cascade.iterrows():
            vertices.add(row_data.tid)

            # This means they are the root, and don't have a parent
            if row_data.parent_tid == "-1":
                continue

            edge_list.append((row_data.parent_tid, row_data.tid))

        # Create graph, add vertices, add edges
        g = Graph(directed=True)
        g.cascade_id = cascade_id
        g.add_vertices(list(vertices))
        g.add_edges(edge_list)

        # Store graph in list
        graph_list.append(g)

        if len(graph_list) % 10_000 == 0:
            # Generate zfilled file name
            fname = f"time_inferred_diffusion_cascades_{str(cas_file_num).zfill(2)}.pkl"
            print(f"\t- Saving {fname}...")
            output_path_fname = os.path.join(outdir, fname)
            with open(output_path_fname, "wb") as pickle_file:
                pkl.dump(graph_list, pickle_file)
            cas_file_num += 1
            graph_list = []

    # Save the last version which didn't hit the length above to save the file
    fname = f"time_inferred_diffusion_cascades_{str(cas_file_num).zfill(2)}.pkl"
    print(f"\t- Saving {fname}...")
    output_path_fname = os.path.join(outdir, fname)
    with open(output_path_fname, "wb") as pickle_file:
        pkl.dump(graph_list, pickle_file)

    print("Script complete.")
