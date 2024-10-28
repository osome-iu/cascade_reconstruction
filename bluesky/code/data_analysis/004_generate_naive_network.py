"""
Purpose: Create the naive reshare network based on the sampled Bluesky cascade data.

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
SAMPLED_CASCADES_FILE = "../../data/sampled_cascades/sampled_cascades.parquet"
OUTPUT_DIR = "/data_volume/cascade_reconstruction/bluesky_networks"
CASCADES_DIR = "/data_volume/cascade_reconstruction/pdi_bluesky/gamma_0_75/alpha_1_1"
URIS_FILE = (
    "/data_volume/cascade_reconstruction/bluesky/sampled_cascades/final_uris_list.txt"
)


def generate_naive_network(df, uris_list):
    """
    Generate a naive directed and weighted retweet network based on the provided df.

    Parameters
    -----------
    - df (pandas dataframe): dataframe containing the sampled reshare data
    - uris_list (list[str]): a list of the uris that we use in the actual networks after
        excluding weird edge cases. E.g., cascades that have reshares with timestampes that
        occur *before* the original post. This is a potential problem with Bluesky data as
        any timestamp can be utilized.

    Returns
    -----------
    global_net (igraph.Graph): a naive directed and weighted retweet network
    """
    # Select only the cascades we want
    original_posts = df[df.uri.isin(uris_list)]
    reshared_posts = df[df.subject_uri.isin(uris_list)]
    df = pd.concat([original_posts, reshared_posts])

    all_vertices = set(df["author"])
    all_edges = list()

    # Iterate over unique cascade_ids
    for cascade_id in uris_list:
        # Filter the DataFrame for the current cascade_id (uri)
        cas_original_posts = df[df.uri == cascade_id]
        cas_reshared_posts = df[df.subject_uri == cascade_id]
        cascade_df = pd.concat([cas_original_posts, cas_reshared_posts])

        # Extract the root and retweeters user_ids
        is_root = cascade_df["type"] == "post"
        root = cascade_df.loc[is_root, "author"].values[0]
        retweeters = cascade_df.loc[~is_root, "author"].values

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

    print("Collecting the good URIs we used...")
    with open(URIS_FILE, "r") as f:
        uris_list = [l.rstrip() for l in f]

    print("Loading sampled cascade data...")
    df = pd.read_parquet(SAMPLED_CASCADES_FILE)

    print("Generating the network...")
    global_net = generate_naive_network(df, uris_list)

    print("Saving the network...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"naive_network.gmlz")
    global_net.write_graphmlz(output_path)

    print("--- Script complete ---")
