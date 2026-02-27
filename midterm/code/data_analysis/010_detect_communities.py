"""
Purpose:
    This script detects communities in network files using the Louvain method. It processes a
    naive network and multiple reconstructed network versions with different gamma and alpha
    parameters. The script performs multiple Louvain runs on each network, calculates modularity
    and Jaccard similarity indices, and saves the results.

    We calculate jaccard similarities for the following cases:
        - Naive network vs. itself 100x vs. the first Louvain run
        - Naive network vs. each reconstructed network version 100x vs. the first Louvain run

Inputs:
    Naive network file (naive_network.gmlz)
    Reconstructed network files in subdirectories defined by gamma and alpha parameters

Outputs:
    Parquet file with modularity values for each network version. Columns are:
        - modularity (float): modularity value
        - net_version (str): network version number (None for the naive network)
        - gamma (float): gamma parameter from PDI (None for the naive network)
        - alpha (float): alpha parameter PDI (None for the naive network)
    Parquet file with Jaccard similarity indices comparing each network versions.
    Columns are:
        - jaccard_sim (float): Jaccard similarity index
        - net_version (str): network version number (None for the naive network)
        - gamma (float): gamma parameter from PDI (None for the naive network)
        - alpha (float): alpha parameter PDI (None for the naive network)
"""

import glob

import os

import igraph as ig

import pandas as pd

from clusim.clustering import Clustering
import clusim.sim as sim

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set input directories
NETWORKS_DIR = "/data_volume/cascade_reconstruction/midterm_networks"
GAMMA_DIRS = ["gamma_0_5", "gamma_0_25", "gamma_0_75"]
ALPHA_DIRS = ["alpha_1_1", "alpha_2_0", "alpha_3_0"]

# Set up output directory
COMMUNITIES_DIR = "/data_volume/cascade_reconstruction/networks_stats/communities"
os.makedirs(COMMUNITIES_DIR, exist_ok=True)

# Analysis info
LOUVAIN_REPS = 100


def convert_dir_name_to_numeric(dir_name, type=None):
    """
    Convert the gamma/alpha dirs to the numeric number they represent.

    Parameters:
    -----------
    dir_name (str): directory name

    Return:
    ----------
    float
    """
    options = ["gamma", "alpha"]
    if type not in options:
        raise TypeError(f"`type` must be one of {options}")
    dir_tail = dir_name.replace(f"{type}_", "")
    dir_chunks = dir_tail.split("_")
    return float(".".join(dir_chunks))


def get_file_path_info(networks_dir, gamma_dirs, alpha_dirs):
    """
    Create a list of all full paths to all network files and associated network info.

    Parameters:
    -----------
    - networks_dir (str) : path to directory where network files are saved
    - gamma_dirs (list[str]) : a list of subdirectories that exist within
        `networks_dir`
    - alpha_dirs (list[str]) : a list of subdirectories that exist within
        each item within `gamma_dirs`

    Returns:
    ----------
    - file_paths (list[dict]) : a list of dictionary objects that contain the file
        path for each network version along with the gamma and alpha settings for their
        generation. Ex:
        {
            "net_version" : "001",  # None for the naive version
            "gamma" : 0.5,  # None for the naive version
            "alpha" : 1.1,  # None for the naive version
            "file_path" : "path/to/file.gmlz"
        }
    """

    naive_network_fp = os.path.join(networks_dir, "naive_network.gmlz")
    if not os.path.exists(naive_network_fp):
        raise Exception("`naive_network_fp` path does not exist!")

    # We will need gamma/alpha later, but include "naive" here for this separate case
    file_paths = [
        {
            "net_version": None,
            "gamma": None,
            "alpha": None,
            "file_path": naive_network_fp,
        }
    ]

    for gamma_dir in gamma_dirs:
        gamma_numeric = convert_dir_name_to_numeric(gamma_dir, "gamma")

        for alpha_dir in alpha_dirs:
            alpha_numeric = convert_dir_name_to_numeric(alpha_dir, "alpha")

            fps = glob.glob(
                os.path.join(
                    networks_dir, gamma_dir, alpha_dir, "network_version_*.gmlz"
                )
            )
            sorted_fps = sorted(fps)

            for fp in sorted_fps:
                # Extract the network version number. Ex name: network_version_001.gmlz
                basename = os.path.basename(fp)
                stripped_name = basename.replace("network_version_", "")
                net_version = stripped_name.replace(".gmlz", "")
                file_paths.append(
                    {
                        "net_version": net_version,
                        "gamma": gamma_numeric,
                        "alpha": alpha_numeric,
                        "file_path": fp,
                    }
                )

    return file_paths


def get_louvain_partition(g, resolution=1):
    """
    Get the Louvain partition for a given graph.

    Parameters:
    -----------
    - g (igraph.Graph) : an igraph graph object
    - resolution (float) : resolution parameter for the Louvain algorithm

    Returns:
    ----------
    - louvain_partition (igraph.clustering.VertexClustering) : a VertexClustering object
    """
    # If directed, throw error
    if g.is_directed():
        raise ValueError("Graph must be undirected!")

    # Find partitions:
    #   Smaller resolution values result in a smaller number of larger clusters,
    #   while higher values yield a large number of small clusters. The classical
    #   modularity measure assumes a resolution parameter of 1 so we use that.
    return g.community_multilevel(
        weights=g.es["weight"], resolution=resolution, return_levels=False
    )


def create_clustering_object(g, louvain_partition):
    """
    Create a Clustering object from an igraph VertexClustering object.

    Parameters:
    -----------
    - g (igraph.Graph) : an igraph graph object
    - louvain_partition (igraph.clustering.VertexClustering) : a VertexClustering object

    Returns:
    ----------
    - clustering (clusim.clustering.Clustering) : a Clustering object
    """
    # Create clustering object
    element_2_cluster = dict()
    for node, membership in zip(g.vs, louvain_partition.membership):

        # We need to use the node "name" to ensure that we are comparing
        # nodes correctly across clusterings
        node_id = node["name"]
        element_2_cluster[node_id] = [membership]

    return Clustering(elm2clu_dict=element_2_cluster)


if __name__ == "__main__":
    # Collect all file paths and their associated parameters
    file_path_info_list = get_file_path_info(
        networks_dir=NETWORKS_DIR, gamma_dirs=GAMMA_DIRS, alpha_dirs=ALPHA_DIRS
    )

    # Results stored here
    modularities = []
    jaccard_similarities = []

    # Extract the naive file path info
    naive_file_path_info = file_path_info_list.pop(0)

    # First we compare the naive network to itself after multiple Louvain runs
    print(f"Begin {LOUVAIN_REPS} naive comparisons...")
    G = ig.Graph.Read_GraphMLz(naive_file_path_info["file_path"])

    # Drop directionality for Louvain
    # "each" preserves the edge weights while removing directionalty
    G.to_undirected(mode="each")

    louvain_partition = get_louvain_partition(G)

    modularities.append(
        {
            "modularity": louvain_partition.modularity,
            "net_version": naive_file_path_info["net_version"],
            "gamma": naive_file_path_info["gamma"],
            "alpha": naive_file_path_info["alpha"],
        }
    )

    # Create clustering object
    naive_clustering = create_clustering_object(G, louvain_partition)

    # Repeat the louvain process for the naive network and make clustering comparisons
    for n in range(LOUVAIN_REPS):
        louvain_partition = get_louvain_partition(G)

        # Store modularity
        modularities.append(
            {
                "modularity": louvain_partition.modularity,
                "net_version": naive_file_path_info["net_version"],
                "gamma": naive_file_path_info["gamma"],
                "alpha": naive_file_path_info["alpha"],
            }
        )

        # Create clustering object, compare to the original and save
        clustering = create_clustering_object(G, louvain_partition)
        j_sim = sim.jaccard_index(clustering, naive_clustering)
        jaccard_similarities.append(
            {
                "jaccard_sim": j_sim,
                "net_version": naive_file_path_info["net_version"],
                "gamma": naive_file_path_info["gamma"],
                "alpha": naive_file_path_info["alpha"],
            }
        )

    num_files = len(file_path_info_list)
    print(f"Begin comparing naive network to {num_files} reconstructed versions...")
    for fnum, fp_info in enumerate(file_path_info_list, start=1):
        fname = fp_info["file_path"]
        print(f"Working on {fname} ({fnum}/{num_files})")
        G = ig.Graph.Read_GraphMLz(fp_info["file_path"])
        G.to_undirected(mode="each")

        for n in range(LOUVAIN_REPS):

            louvain_partition = get_louvain_partition(G)

            # Store modularity
            modularities.append(
                {
                    "modularity": louvain_partition.modularity,
                    "net_version": fp_info["net_version"],
                    "gamma": fp_info["gamma"],
                    "alpha": fp_info["alpha"],
                }
            )

            # Create clustering object, compare to the naive and save
            clustering = create_clustering_object(G, louvain_partition)
            j_sim = sim.jaccard_index(clustering, naive_clustering)
            jaccard_similarities.append(
                {
                    "jaccard_sim": j_sim,
                    "net_version": fp_info["net_version"],
                    "gamma": fp_info["gamma"],
                    "alpha": fp_info["alpha"],
                }
            )

    modularity_df = pd.DataFrame.from_records(modularities)
    jaccard_df = pd.DataFrame.from_records(jaccard_similarities)

    modularity_out_fname = f"modularities.parquet"
    j_sim_out_fname = f"jaccard_similarities.parquet"

    modularity_outpath = os.path.join(COMMUNITIES_DIR, modularity_out_fname)
    modularity_df.to_parquet(modularity_outpath, index=False)

    j_sim_outpath = os.path.join(COMMUNITIES_DIR, j_sim_out_fname)
    jaccard_df.to_parquet(j_sim_outpath, index=False)
