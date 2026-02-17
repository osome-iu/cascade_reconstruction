"""
Purpose:
    Reconstruct all cascades randomly.

Note:
    We do not generate multiple versions of cascades with length = 2 because
    there is no guess work. Given the temporal ordering, there is only one
    potential parent so we generate only one version.

Input:
    None. Paths are set and data is loaded in.

Output:
    - Cascades are saved in: output/reconstructed_data/random/
        - The directory will contain directory names based on cascade ID numbers that have
        been "zfilled" for sorting purposes. Inside each directory will be multiple files,
        each containing a GMLz file representing one version of that cascade reconstructed.
    - E.g., the structure below shows 100 versions of cascade "1":
        ```
        `00001/`
        |- v_001.gml
        |- v_002.gml
        |- ...
        |- v_100.gml
        ```
Authors:
- Francesco Pierri
- Matthew DeVerna
"""
import multiprocessing
import os
import time
import warnings

import numpy as np
import pandas as pd

from igraph import Graph
from joblib import Parallel, delayed


N_SIMULATIONS = 100
N_PROCS = multiprocessing.cpu_count()

DATA_DIR = "../../cleaned_data"
OUT_DIR = "../../output/reconstructed_data/random"


def generate_random_reconstruction(poten_edge_users):
    """
    Generate a random reconstruction of potential edge users.

    Parameters:
    -----------
    - poten_edge_users: A temporally ordered list of potential edge users

    Returns:
    -------
    - edge_list: A list of tuples representing the edges in the random reconstruction
    """
    # The first edge is known, given the temporal ordering
    edge_list = [(poten_edge_users[0], poten_edge_users[1])]

    # Start at the third node
    for idx in range(2, len(poten_edge_users)):
        selected_parent_node = np.random.choice(poten_edge_users[:idx])
        current_node = poten_edge_users[idx]
        edge_list.append((selected_parent_node, current_node))

    return edge_list


def create_output_fname(cascade_id, out_dir, version_num):
    """
    Create an output file path. Z-fill the version number if necessary.

    Parameters:
    ----------
    - cascade_id (str) : the ID of the cascade
    - out_dir (str) : the output directory
    - version_num (int) : the version number of the output file

    Returns:
    ----------
    - output_fname (str) : the output file path
    """
    output_dir = os.path.join(out_dir, cascade_id.zfill(5))
    os.makedirs(output_dir, exist_ok=True)
    output_fname = os.path.join(output_dir, f"v_{str(version_num).zfill(3)}.gmlz")
    return output_fname


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data = pd.read_parquet(os.path.join(DATA_DIR, "clean_raw_data_anon.parquet"))

    # Converting tweet_date to datetime (and sorting by cascade_id just for sake of aesthetics)
    data["tweet_date"] = data.tweet_date.dt.to_pydatetime()
    data.sort_values("cascade_id", ascending=True, inplace=True)

    all_cascades = list(data["cascade_id"].unique())
    num_cascades = len(all_cascades)

    try:
        timer_records = []
        for cas_num, cascade_id in enumerate(all_cascades, start=1):
            print(f"Working on cascade: {cascade_id} ({cas_num}/{num_cascades})...")

            start = time.time()

            # Select a single cascades data
            cascade_frame = data[data["cascade_id"] == cascade_id]

            cascade_frame = cascade_frame.sort_values("tweet_date", ascending=True)

            num_tweets = len(cascade_frame)
            print(f"\t - Number of tweets: {num_tweets:,}")

            # We only need to generate one version of cascades len 2 because we are not guessing.
            num_cascade_versions = 1 if (num_tweets == 2) else N_SIMULATIONS

            # Create lists for the function below
            poten_edge_users = list(cascade_frame["tid"])

            ## Using joblib (https://joblib.readthedocs.io/en/latest/) ----> much faster
            backend = "loky"
            edge_lists = Parallel(n_jobs=N_PROCS, backend=backend)(
                delayed(generate_random_reconstruction)(poten_edge_users)
                for _ in range(num_cascade_versions)
            )

            # Save each version as it's own graph file
            for vnum, edge_list in enumerate(edge_lists, start=1):
                g = Graph(directed=True)

                # Generate vertex set
                all_vertices = set()
                for source, target in edge_list:
                    all_vertices.add(source)
                    all_vertices.add(target)

                # Add vertices and edges
                g.add_vertices(list(all_vertices))
                g.add_edges(edge_list)

                # Create the correct directory and filename
                output_path_fname = create_output_fname(
                    cascade_id=cascade_id,
                    out_dir=OUT_DIR,
                    version_num=vnum,
                )

                # Write the compressed graphml file for each cascade version
                g.write(output_path_fname, format="graphmlz")

            elapsed_time = time.time() - start
            minutes, seconds = divmod(elapsed_time, 60)
            print(f"\t - Time: {minutes}m {seconds}s [{elapsed_time}]")
            timer_records.append(
                {
                    "cascade_id": cascade_id,
                    "cascade_length": num_tweets,
                    "num_seconds": elapsed_time,
                }
            )
    # Allow for keyboard interrupt to get a sense of timing...
    except KeyboardInterrupt:
        timer_records_df = pd.DataFrame.from_records(timer_records)
        timer_records_df.to_parquet(os.path.join(OUT_DIR, "timer_records.parquet"))
    finally:
        timer_records_df = pd.DataFrame.from_records(timer_records)
        timer_records_df.to_parquet(os.path.join(OUT_DIR, "timer_records.parquet"))

print("#" * 50)
print(f"Script complete!")
