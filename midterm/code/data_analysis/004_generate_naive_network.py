"""
Purpose: Create the naive retweet network based on the sampled cascade data.

Input: None
    - Reads data files based on hardcoded path (see constants below)

Output: One naive retweet network
    - Filename form: `naive_network.gmlz`

Author: Matthew DeVerna
"""

import os

import numpy as np
import pandas as pd

from collections import Counter
from igraph import Graph


# Constants
NUM_NET_VERSIONS = 100
SAMPLED_CASCADES_FILE = "../../data/sampled_cascades_records/cascade_records.parquet"
OUTPUT_DIR = "/data_volume/cascade_reconstruction/midterm_networks/"


def generate_naive_network(df):
    """
    Generate a naive directed and weighted retweet network based on the provided df.

    Parameters
    -----------
    - df (pandas dataframe): dataframe containing the sampled retweet data

    Returns
    -----------
    global_net (igraph.Graph): a naive directed and weighted retweet network
    """
    all_vertices = set(df["user_id"])
    all_edges = list()

    # Iterate over unique cascade_ids
    for tweet_id in df["cascade_id"].unique():
        # Filter the DataFrame for the current cascade_id
        cascade_df = df[df["cascade_id"] == tweet_id]

        # Extract the root and retweeters user_ids
        is_root = cascade_df["is_root"]
        root = cascade_df.loc[is_root, "user_id"].values[0]
        retweeters = cascade_df.loc[~is_root, "user_id"].values

        # Use broadcasting to create pairs
        new_edges = np.column_stack((np.full(retweeters.shape, root), retweeters))

        # Flatten the array of arrays into a list of tuples and extend the edges list
        all_edges.extend(map(tuple, new_edges))

    # Create the global network and add the vertices and edges
    global_net = Graph(directed=True)
    global_net.add_vertices(list(all_vertices))
    global_net.add_edges(list(set(all_edges)))

    # Determine edge weights
    edge_weight_counter = Counter()
    for edge in all_edges:
        edge_weight_counter[edge] += 1

    # Update weights
    global_net.es["weight"] = [
        edge_weight_counter[
            (global_net.vs[edge.source]["name"], global_net.vs[edge.target]["name"])
        ]
        for edge in global_net.es
    ]

    return global_net


if __name__ == "__main__":

    print("Loading sampled cascade data...")
    df = pd.read_parquet(SAMPLED_CASCADES_FILE)

    print("Generating the network...")
    global_net = generate_naive_network(df)

    print("Saving the network...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"naive_network.gmlz")
    global_net.write_graphmlz(output_path)

    print("--- Script complete ---")
