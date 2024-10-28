"""
Purpose:
    Reconstruct all cascades using PDI.

Note:
    - Cascades of length 2 are not reconstructed as they are not cascades.
    - Cascades are saved in directories that indicate their creation parameters.
        - This information is also included within each cascade object itself.

Input:
    None. Paths are set and data is loaded in.

Output:
    - Cascades out output to the output/reconstructed_data directory.
        - The output directory will contain subdirectories that indicate the creation parameters
            utilized (gamma and alpha).
        - Inside those subdirectories, a directory will be created based on the cascade ID
        - Files inside the cascade ID directory (GMLz format) will represent different versions
            of that cascade ID and will be zfilled and numbered 001, 002, ... 100.
    - E.g., the structure below shows 100 versions of cascade 1:
        ```
        `00001/`
        |- v_001.gmlz
        |- v_002.gmlz
        |- ...
        |- v_100.gmlz
        ```
Authors:
- Francesco Pierri
- Matthew DeVerna
"""

import argparse
import multiprocessing
import os
import warnings

import numpy as np
import pandas as pd

from igraph import Graph
from joblib import Parallel, delayed

# Local project package
from pkg import reconstruction

N_SIMULATIONS = 100
N_PROCS = multiprocessing.cpu_count()

DATA_DIR = "../../cleaned_data"
OUT_DIR = "/data_volume/cascade_reconstruction"

# Setting values for the power law
GAMMA_VALUES = [0.25, 0.5, 0.75]
ALPHA_VALUES = [1.1, 1.5, 2.0, 2.5, 3.0]
XMIN = 1


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Reconstruct all cascades using PDI.")
    parser.add_argument(
        "--type",
        required=True,
        choices=["midterm", "vosoughi"],
        help="Specify the type of cascade you are reconstructing: [midterm, vosoughi]",
    )
    args = parser.parse_args()
    return args


def entire_cascade_reconstruction(
    poten_edge_users, poten_edge_tstamps, poten_edge_fcounts, gamma, alpha, xmin
):
    """
    Function to reconstruct a single cascade using pkg library.

    Parameters:
    ----------
    - Same as reconstruction.get_who_rtd_whom() from local package.

    Returns:
    ----------
    - edge_list (list) : list of tuples representing a directed graph's edges
        - Format will be (source, target) or (parent_tid, retweeter_tid)
    """
    # Must be true so we save the calculation below
    edge_list = [(poten_edge_users[0], poten_edge_users[1])]

    # Iterate over all retweets (but the root and first RT) to infer the parent
    num_retweets = len(poten_edge_tstamps)
    for idx in range(2, num_retweets):
        # Get the timestamp and tweet ID for the retweeter
        retweeter_tstamp = poten_edge_tstamps[idx]
        retweeter_tid = poten_edge_users[idx]

        # Extract lists related to the potential parents
        potential_tstamps = np.array(poten_edge_tstamps[:idx])
        potential_parents = poten_edge_users[:idx]  # These are tweet IDs
        potential_parents_fcounts = poten_edge_fcounts[:idx]

        # Infer parent
        inferred_parent = reconstruction.get_who_rtd_whom(
            poten_edge_users=potential_parents,
            poten_edge_tstamps=potential_tstamps,
            poten_edge_fcounts=potential_parents_fcounts,
            curr_tstamp=retweeter_tstamp,
            gamma=gamma,
            alpha=alpha,  # can be determined by the fitting
            xmin=xmin,  # this should be 1, representing one second
        )

        # Should be (source, target)
        edge_list.append((inferred_parent, retweeter_tid))
    return edge_list


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Set output directory based on cascade type
    args = parse_args()
    cascade_type = f"pdi_{args.type}"
    if cascade_type == "pdi_midterm":
        raise Exception("Not implemented! Handled in a separate script!")

    out_dir = os.path.join(OUT_DIR, cascade_type)

    # Load data
    data = pd.read_parquet(os.path.join(DATA_DIR, "clean_raw_data_anon.parquet"))

    # Converting tweet_date to datetime (and sorting by cascade_id just for sake of aesthetics)
    data["tweet_date"] = data.tweet_date.dt.to_pydatetime()
    data.sort_values("cascade_id", ascending=True, inplace=True)

    all_cascades = list(data["cascade_id"].unique())
    num_cascades = len(all_cascades)

    for gamma in GAMMA_VALUES:
        gamma_dir = os.path.join(out_dir, f"gamma_{gamma}".replace(".", "_"))

        print("#" * 50)
        print(f"gamma = {gamma}".upper())
        print("#" * 50)

        for alpha in ALPHA_VALUES:
            alpha_dir = os.path.join(gamma_dir, f"alpha_{alpha}".replace(".", "_"))

            print("#" * 50)
            print(f"alpha = {alpha}".upper())
            print("#" * 50)

            # Skip completed alpha levels
            if os.path.exists(alpha_dir):
                num_cascade_dirs = len(os.listdir(alpha_dir))
                if num_cascade_dirs == num_cascades:
                    print(f"We are reconstructing {num_cascades} cascades.")
                    print(f"Cascade directories in {alpha_dir}: {num_cascade_dirs}")
                    print("All cascades generated. Skipping...")
                    continue
                else:
                    print(f"We are reconstructing {num_cascades} cascades.")
                    print(f"Cascade directories in {alpha_dir}: {num_cascade_dirs}")
                    print("Some cascades not generated so we continue...")

            for cas_num, cascade_id in enumerate(all_cascades, start=1):
                print(f"Working on cascade: {cascade_id} ({cas_num}/{num_cascades})...")
                cascade_dir = os.path.join(alpha_dir, cascade_id.zfill(5))
                if os.path.exists(cascade_dir) and len(os.listdir(cascade_dir)) == 100:
                    print(f"\t - Directory <{cascade_dir}> already exists. Skipping...")
                    continue
                os.makedirs(cascade_dir, exist_ok=True)

                # Select a single cascades data
                cascade_frame = data[data["cascade_id"] == cascade_id]

                cascade_frame = cascade_frame.sort_values("tweet_date", ascending=True)

                num_tweets = len(cascade_frame)
                print(f"\t - Number of tweets: {num_tweets:,}")

                # We only need to generate one version of cascades len 2 because we are not guessing.
                num_cascade_versions = 1 if (num_tweets == 2) else N_SIMULATIONS

                # Create lists for the function below
                poten_edge_users = list(cascade_frame["tid"])
                poten_edge_tstamps = [
                    int(dt.timestamp()) for dt in cascade_frame.tweet_date
                ]
                poten_edge_fcounts = np.array(cascade_frame["user_followers"])

                ## Using joblib (https://joblib.readthedocs.io/en/latest/) ----> much faster
                backend = "loky"
                edge_lists = Parallel(n_jobs=N_PROCS, backend=backend)(
                    delayed(entire_cascade_reconstruction)(
                        poten_edge_users,
                        poten_edge_tstamps,
                        poten_edge_fcounts,
                        gamma,
                        alpha,
                        XMIN,
                    )
                    for _ in range(num_cascade_versions)
                )

                # Save each version as it's own graph file
                for vnum, edge_list in enumerate(edge_lists, start=1):
                    # Create the graph, store cascade ID and creation parameters
                    g = Graph(directed=True)
                    g.cascade_id = cascade_id
                    g.alpha = alpha
                    g.gamma = gamma

                    # Generate vertex set
                    all_vertices = set()
                    for source, target in edge_list:
                        all_vertices.add(source)
                        all_vertices.add(target)

                    # Add vertices and edges
                    g.add_vertices(list(all_vertices))
                    g.add_edges(edge_list)

                    # Create the correct directory and filename
                    output_path_fname = os.path.join(
                        cascade_dir, f"v_{str(vnum).zfill(3)}.gmlz"
                    )

                    # Write the compressed graphml file for each cascade version
                    g.write(output_path_fname, format="graphmlz")

    print("#" * 50)
    print(f"Script complete!")
