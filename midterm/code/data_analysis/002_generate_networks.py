"""
Purpose: Create multiple versions of the retweet network from
    the PDI versions of cascades that we have generated.

Approach:
    - We create versions of the full network by putting together all version 1
        cascades, all version 2 cascades, and so on.
    - Note that each cascade has either one or one hundred versions.
    - Only one verson is created if the cascade has a length of 2, since we can
        infer the true cascade perfectly (based on timestamps).
    - As a result, for cascades with only one version, that version is utilized in
        all versions of the network.
    - Out of our 10k sampled cascades, 5,004 have length > 2 (have 100 versions),
        and 4,996 have length = 2 (have only one version).

Input: None
    - Reads data files based on hardcoded path (see constants below)

Output: Multiple versions of the retweet network
    - Filename form: `network_version_{n}.gmlz`

Author: Matthew DeVerna
"""
import os

from collections import defaultdict
from collections import Counter

from igraph import Graph
from joblib import Parallel, delayed


# Constants
NUM_NET_VERSIONS = 100
DATA_DIR = "/data_volume/cascade_reconstruction/pdi_midterm/gamma_0_25/"
ALPHA_DIRS = ["alpha_1_1", "alpha_2_0", "alpha_3_0"]  # Sub dirs of DATA_DIR
OUTPUT_DIR = "/data_volume/cascade_reconstruction/midterm_networks/"


def generate_cascade_numv_map(data_dir):
    """
    Generate a dictionary mapping cascade id to

    Paramters
    ----------
    - data_dir (str): the path to the directory that contains the pdi cascade versions

    Returns
    ---------
    - cascade_numv_map (dict): maps the cascade ID (str) to the number of versions
    """

    cascade_numv_map = dict()

    for cascade_id in os.listdir(data_dir):
        full_path = os.path.join(data_dir, cascade_id)
        cas_files = os.listdir(full_path)
        cascade_numv_map[cascade_id] = len(cas_files)

    return cascade_numv_map


def generate_net_ver_paths_map(cascade_numv_map, data_dir, n_versions):
    """
    Generate a dictionary mapping the global network version number
    to the list of paths to use to generate that version.

    Parameters
    ----------
    - cascade_numv_map (dict): maps the cascade ID (str) to the number
        of versions that we have of that cascade. Either 100 or 1.
    - data_dir (str): the path to the directory that contains the pdi cascade versions
    - n_versions (int): the number of networks to generate.

    Returns
    ---------
    - net_ver_files_map (dict): maps the network version to a list of
        file paths that will be utilized to generate that verison of
        the network
    """

    net_ver_files_map = defaultdict(list)
    for ver in range(1, n_versions + 1):
        for cascade_id, num_files in cascade_numv_map.items():
            # If we have 100 versions, use the specified version
            if num_files == 100:
                padded_ver = f"{str(ver)}".zfill(3)
                net_ver_files_map[ver].append(
                    os.path.join(data_dir, cascade_id, f"v_{padded_ver}.gmlz")
                )

            # Otherwise, there is only 1 version (cascade length = 2)
            # so we use that version specifically
            else:
                net_ver_files_map[ver].append(
                    os.path.join(data_dir, cascade_id, "v_001.gmlz")
                )
    return dict(net_ver_files_map)


def generate_network(files_to_load, netv):
    """
    Generate global network.

    Parameters
    ----------
    - file_to_load (list[str]): paths to files of cascades that will be combined
        to create the global network
    - netv (int) : the version of the network (added to the network meta data)

    Returns
    ---------
    - global_net (igraph.Graph): Directed retweet network inclusive of all cascades
    """
    # Load all verticies and edges.
    # Use "names" (user IDs) to be consistent across cascades
    all_vertices = set()
    all_edges = list()
    for file in files_to_load:
        g = Graph.Read_GraphMLz(file)
        all_vertices.update(g.vs["name"])
        all_edges.extend(
            [(g.vs[edge.source]["name"], g.vs[edge.target]["name"]) for edge in g.es]
        )

    # Create the global network and add the vertices and edges
    global_net = Graph(directed=True)
    global_net["version"] = netv
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
    for alpha_dir in ALPHA_DIRS:
        print(f"Begin: {alpha_dir}")

        out_dir = os.path.join(OUTPUT_DIR, alpha_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Skip if all networks have been generated
        if len(os.listdir(out_dir)) == NUM_NET_VERSIONS:
            print(f"\t - All networks have been generated. Skipping...")
            continue

        print("Mapping cascade IDs to number of versions...")
        full_data_path = os.path.join(DATA_DIR, alpha_dir)
        cascade_numv_map = generate_cascade_numv_map(full_data_path)

        print("Mapping global network version number to cascade files to load...")
        net_ver_files_map = generate_net_ver_paths_map(
            cascade_numv_map, full_data_path, NUM_NET_VERSIONS
        )

        print("Beginning network generation...")

        def process_version(netv):
            v_str = str(netv).zfill(3)
            output_path = os.path.join(out_dir, f"network_version_{v_str}.gmlz")
            if not os.path.exists(output_path):
                files_to_load = net_ver_files_map[netv]
                global_net = generate_network(files_to_load, netv)
                global_net.write_graphmlz(output_path)
                print(f"\t - Generated version {netv}.")

        # Parallel execution
        Parallel(n_jobs=-1)(
            delayed(process_version)(netv) for netv in range(1, NUM_NET_VERSIONS + 1)
        )

    print("--- Script complete ---")
