"""
Purpose:
    Calculate node centralities for the naive network.

Inputs:
    None, network loaded with constant below

Output:
    - .parquet files for each network version
    - Rows represent nodes and columns are:
        - "user_id": user id (node "name")
        - "degree": node out degree
        - "strength": node out strength
        - "kcore": node kcore (based on out degree),
        - "eigenval": node eigenvector centrality (weighted)

Authors: 
- Matthew DeVerna
- Francesco Pierri
"""

import os
import pandas as pd
import warnings

# Suppress specific RuntimeWarning (for eigenvector centrality)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from igraph import Graph


NAIVE_NET_FILE = (
    "/data_volume/cascade_reconstruction/midterm_networks/naive_network.gmlz"
)
OUTPUT_DIR = "/data_volume/cascade_reconstruction/networks_stats/centralities"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load network and calculate centralities
graph = Graph.Read_GraphMLz(NAIVE_NET_FILE)
weights = graph.es["weight"]

# Simple centralities
degree = graph.degree(mode="out")
strength = graph.strength(mode="out", weights=weights)

# k-core
coreness = graph.coreness(mode="out")

# NOTE:
#       I comment this out because it takes forever.
#       Can include later, if we want.
# Betweenness
# distances = [1 / w for w in weights]
# betweenness = graph.betweenness(weights=distances)

# Eigenvector centrality
eigens = graph.eigenvector_centrality(weights=weights)

# Build dataframe
centrality_records = []
data_zip = zip(list(graph.vs), degree, strength, coreness, eigens)
for node, degr, stren, kcore, eig in data_zip:
    centrality_records.append(
        {
            "user_id": node["name"],
            "degree": degr,
            "strength": stren,
            "kcore": kcore,
            "eigenval": eig,
        }
    )
df = pd.DataFrame.from_records(centrality_records)

# Save df
fname = f"naive_network_centralities.parquet"
outpath = os.path.join(OUTPUT_DIR, fname)
df.to_parquet(outpath, index=False, engine="pyarrow")
