"""
Purpose:
    Compute the homophily in a network based on Cinelli et al. PNAS (2021), 
    i.e., by leveraging political score of users (node score vs average neighbors score)

Note:
    We filter out all nodes in the network that do not have a political score.

Inputs:
    .gmlz files in data dir specified by constants (see below)
    
Outputs:
    .parquet files in data dir specified by constants (see below)
     - Rows represent users and Columns are:
        - "user_id": user 
        - "score": political score of the user
        - "avg_neigh_score": average political score of its neighbors
        - "avg_w_neigh_score": average political score of its neighbors (weighted by retweets)

Authors:
    Francesco Pierri
"""

import os

import networkx as nx
import numpy as np
import pandas as pd

from midterm.utils import collect_files_recursively

EDGELISTS_DIR = "/data_volume/cascade_reconstruction/edgelists"
POL_SCORE_FILE = "../../data/user_political_score.parquet"
OUTPUT_DIR = "/data_volume/cascade_reconstruction/networks_stats/homophily"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MATCHING_STR = "*.parquet"

if __name__ == "__main__":

    # Load files
    files = collect_files_recursively(matching_str=MATCHING_STR, dirname=EDGELISTS_DIR)
    pol_score_df = pd.read_parquet(POL_SCORE_FILE)

    pol_score_dict = {
        row["user_id"]: row["political_score"] for _, row in pol_score_df.iterrows()
    }

    for file_path in sorted(files):

        print(f"Working on:\n\t- {file_path}")
        df = pd.read_parquet(file_path)

        # Filtering outnodes in the network that don't have a political score
        df = df[df["source"].isin(pol_score_dict) & df["target"].isin(pol_score_dict)]

        # Read edgelist and convert it to an undirected weighted graph
        g = nx.from_pandas_edgelist(df, edge_attr = "weight", create_using = nx.Graph)

        # Looping over all nodes
        records = []
        for u in g.nodes:
            neighbors_scores = []
            neighbors_weights = []
            
            for edge in g.edges(u):
                v = edge[1] # edge = (u,v)
                if v == u: # ignore self loops
                    continue

                # append the political score of the neighbor
                weight = int(g.edges[edge]["weight"])
                neighbors_scores.append(pol_score_dict[v])
                neighbors_weights.append(weight)

            # if node has at least one neighbor, include it
            if len(neighbors_scores) > 0:
                records.append(
                    {
                        "user_id": u,
                        "user_ideo": pol_score_dict[u],
                        "neighb_mean_ideo": np.average(a=neighbors_scores),
                        "neighb_wtd_mean_ideo": np.average(
                            a=neighbors_scores, weights=neighbors_weights
                        ),
                    }
                )

        user_ideo_w_neighb_ideo = pd.DataFrame.from_records(records)

        # Save the frame
        basename = os.path.basename(file_path)
        fname = basename.replace("edgelist", "homophily")
        outpath = os.path.join(OUTPUT_DIR, fname)
        user_ideo_w_neighb_ideo.to_parquet(outpath, index=False, engine="pyarrow")

    print("--- Script complete ---")
