"""
Purpose:
    Extract edgelists from all midterm networks.

Inputs:
    .gmlz files in data dir specified by constants (see below)
    
Outputs:
     .parquet files in data dir specified by constants (see below)
     - Rows represent edges and Columns are:
        - "source": user being retweeted
        - "target": user retweeting
        - "weight": number of retweets

Authors:
    Francesco Pierri
"""

import os

from igraph import Graph

import pandas as pd

from midterm.utils import collect_files_recursively


NETWORKS_DIR = "/data_volume/cascade_reconstruction/midterm_networks"
OUTPUT_DIR = "/data_volume/cascade_reconstruction/edgelists"
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

        if "naive" not in file_path:
            params = extract_params(file_path)

        # Load network and get edgelist with names (Twitter user id) and weight
        graph = Graph.Read_GraphMLz(file_path)

        edges_df = []
        for e in graph.es:
            source = e.source
            target = e.target
            source_user_id = graph.vs.find(source)["name"]
            target_user_id = graph.vs.find(target)["name"]
            weight = e["weight"]
            edges_df.append(
                {"source": source_user_id, "target": target_user_id, "weight": weight}
            )
        edges_df = pd.DataFrame(edges_df)

        # Basename format: network_version_001.gmlz
        basename = os.path.basename(file_path)

        # Save edgelist dataframe
        if "naive" in file_path:  # handling the naive network file name
            net_ver_str = "edgelist_"
            param_str = "naive_network"
        else:
            version = int(basename.split("_")[-1].split(".")[0])
            net_ver_str = f"edgelist_v_{str(version).zfill(3)}"
            param_str = f"_gamma_{params['gamma']}_alpha_{params['alpha']}"
        extension = f".parquet"
        fname = f"{net_ver_str}{param_str}{extension}"
        outpath = os.path.join(OUTPUT_DIR, fname)
        edges_df.to_parquet(outpath, index=False, engine="pyarrow")
