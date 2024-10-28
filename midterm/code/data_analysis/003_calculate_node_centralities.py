"""
Purpose:
    Calculate node centralities for reconstructed networks.

Inputs:
    None, networks loaded with constants below

Output:
    - .parquet files for each network version
    - Rows represent nodes and columns are:
        - "net_v": the network version
        - "gamma": gamma value to generate network
        - "alpha": alpha value to generate network
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

from midterm.utils import collect_files_recursively


NETWORKS_DIR = "/data_volume/cascade_reconstruction/midterm_networks"
OUTPUT_DIR = "/data_volume/cascade_reconstruction/networks_stats/centralities"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MATCHING_STR = "*.gmlz"


def extract_params(file_path):
    """
    Extract gamma and alpha parameters from full path.

    Example path:
    - ".../midterm_networks/gamma_0_25/alpha_1_1/network_version_001.gmlz"

    Parameters
    ----------
    - file_path (str) : full path to network file

    Return
    ----------
    - params (dict) : dictionary with following form
        {
            "alpha" : float,
            "gamma" : float
        }
    """

    # Split the file_path into components
    path_components = file_path.split("/")

    # Iterate through each component to find gamma and alpha values
    for component in path_components:
        if component.startswith("gamma_"):
            # Extract the gamma value after 'gamma_'
            gamma = float(component.replace("gamma_", "").replace("_", "."))
        elif component.startswith("alpha_"):
            # Extract the alpha value after 'alpha_'
            alpha = float(component.replace("alpha_", "").replace("_", "."))

    # Construct the params dictionary
    params = {"alpha": alpha, "gamma": gamma}

    return params


if __name__ == "__main__":

    files = collect_files_recursively(matching_str=MATCHING_STR, dirname=NETWORKS_DIR)

    for file_path in sorted(files):
        print(f"Working on:\n\t- {file_path}")
        params = extract_params(file_path)

        # Basename format: network_version_001.gmlz
        basename = os.path.basename(file_path)
        version = int(basename.split("_")[-1].split(".")[0])

        # Load network and calculate centralities
        centrality_records = []
        graph = Graph.Read_GraphMLz(file_path)
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
        data_zip = zip(list(graph.vs), degree, strength, coreness, eigens)
        for node, degr, stren, kcore, eig in data_zip:
            centrality_records.append(
                {
                    "net_v": version,
                    "gamma": params["gamma"],
                    "alpha": params["alpha"],
                    "user_id": node["name"],
                    "degree": degr,
                    "strength": stren,
                    "kcore": kcore,
                    "eigenval": eig,
                }
            )
        df = pd.DataFrame.from_records(centrality_records)

        # Save df
        net_ver_str = f"network_v_{str(version).zfill(3)}"
        param_str = f"_gamma_{params['gamma']}_alpha_{params['alpha']}"
        extension = f".parquet"
        fname = f"{net_ver_str}{param_str}{extension}"
        outpath = os.path.join(OUTPUT_DIR, fname)
        df.to_parquet(outpath, index=False, engine="pyarrow")

    print("--- Script complete ---")
