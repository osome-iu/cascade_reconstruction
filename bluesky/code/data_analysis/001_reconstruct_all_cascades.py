"""
Purpose:
    Reconstruct all bluesky cascades using PDI.

Note:
    - Cascades of length 2 are not reconstructed as they are not cascades.
    - Cascades are saved in directories that indicate their creation parameters.
        - This information is also included within each cascade object itself.

Input:
    None. Paths are set and data is loaded in.

Output:
    - Cascades are output to the output/reconstructed_data directory.
        - The output directory will contain subdirectories that indicate the creation parameters
            utilized (gamma and alpha).
        - Inside those subdirectories, a directory will be created based on the cascade ID
        - Files inside the cascade ID directory (GMLz format) will represent different versions
            of that cascade ID and will be zfilled and numbered 001, 002, ... 100.
    - E.g., the structure below shows 100 versions of cascade `cleaned-uri-string` (replaced by
        a "cleaned" URI value; see function below):
        ```
        `cleaned-uri-string/`
        |- v_001.gmlz
        |- v_002.gmlz
        |- ...
        |- v_100.gmlz
        ```
    - We also save a new line delimited .txt file where each line represents one of the URIs that we generate
Authors:
- Francesco Pierri
- Matthew DeVerna
"""

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

AUTHORS_DIR = "/data_volume/cascade_reconstruction/bluesky/author_profiles"
DATA_DIR = "/data_volume/cascade_reconstruction/bluesky/sampled_cascades"
OUT_DIR = "/data_volume/cascade_reconstruction/pdi_bluesky"

# Setting values for the power law
GAMMA_VALUES = [0.25, 0.5, 0.75]
ALPHA_VALUES = [1.1, 1.5, 2.0, 2.5, 3.0]
XMIN = 1


def load_authors_data():
    """
    Return a dataframe of authors meta data
    """
    author_files = [os.path.join(AUTHORS_DIR, f) for f in os.listdir(AUTHORS_DIR)]
    authors = []
    for f in author_files:
        authors.append(pd.read_parquet(f))
    return pd.concat(authors).reset_index(drop=True)


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
        - Format will be (source, target) or (parent_id, reposter_id)
    """
    # Must be true so we save the calculation below
    edge_list = [(poten_edge_users[0], poten_edge_users[1])]

    # Iterate over all reposts (but the root and first RT) to infer the parent
    num_reposts = len(poten_edge_tstamps)
    for idx in range(2, num_reposts):
        # Get the timestamp and post ID for the reposter
        reposter_tstamp = poten_edge_tstamps[idx]
        reposter_id = poten_edge_users[idx]

        # Extract lists related to the potential parents
        potential_tstamps = np.array(poten_edge_tstamps[:idx])
        potential_parents = poten_edge_users[:idx]  # These are post IDs
        potential_parents_fcounts = poten_edge_fcounts[:idx]

        # Infer parent
        inferred_parent = reconstruction.get_who_rtd_whom(
            poten_edge_users=potential_parents,
            poten_edge_tstamps=potential_tstamps,
            poten_edge_fcounts=potential_parents_fcounts,
            curr_tstamp=reposter_tstamp,
            gamma=gamma,
            alpha=alpha,  # can be determined by the fitting
            xmin=xmin,  # this should be 1, representing one second
        )

        # Should be (source, target)
        edge_list.append((inferred_parent, reposter_id))
    return edge_list


def clean_uri_for_fname(uri):
    """
    Remove the repetative text from a uri for a simpler file name

    Converts
        - 'at://did:plc:223yxsx3ifr4vghg36hi3w4a/app.bsky.feed.post/3kmrdg2pfom2f'
    to
        - '223yxsx3ifr4vghg36hi3w4a-3kmrdg2pfom2f'
    """
    return uri.replace("at://did:plc:", "").replace("/app.bsky.feed.post/", "-")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Load data
    data = pd.read_parquet(os.path.join(DATA_DIR, "sampled_cascades.parquet"))
    authors_df = load_authors_data()

    # Converting clean_dt to datetime (and sorting by uri just for sake of aesthetics)
    data["clean_dt"] = data.clean_dt.dt.to_pydatetime()
    data.sort_values("uri", ascending=True, inplace=True)

    # Drop cascades where we don't have author data for all authors.
    merge_df = pd.merge(
        data,
        authors_df[["did", "followers_count"]],
        left_on="author",
        right_on="did",
        how="inner",  # excludes rows where authors are missing
    )
    # Differences in these counts reflect cascades that, in the new frame, have missing author info
    cascade_df_counts = data.subject_uri.value_counts()
    merged_df_counts = merge_df.subject_uri.value_counts()
    difference = cascade_df_counts - merged_df_counts

    # Find those cascades and drop them (n = 271)
    cas_w_missing_authors = difference[difference != 0].index
    rows_w_missing_authors = data.subject_uri.isin(
        cas_w_missing_authors
    ) | data.uri.isin(cas_w_missing_authors)
    data = data[~rows_w_missing_authors].reset_index(drop=True)

    # Merge again to add follower count column
    data = pd.merge(
        data,
        authors_df[["did", "followers_count"]],
        left_on="author",
        right_on="did",
        how="inner",
    )

    # All other posts have type == "repost"
    all_cascades = list(data[data.type == "post"]["uri"].unique())
    num_cascades = len(all_cascades)

    # Stores all of the uris that we use and later saves them
    all_uris = set()

    for gamma in GAMMA_VALUES:
        gamma_dir = os.path.join(OUT_DIR, f"gamma_{gamma}".replace(".", "_"))

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

            for cas_num, uri in enumerate(all_cascades, start=1):
                print(f"Working on cascade: {uri} ({cas_num}/{num_cascades})...")
                uri_clean = clean_uri_for_fname(uri)
                cascade_dir = os.path.join(alpha_dir, uri_clean)
                if os.path.exists(cascade_dir) and len(os.listdir(cascade_dir)) == 100:
                    print(f"\t - Directory <{cascade_dir}> already exists. Skipping...")
                    continue

                # Select a single cascades data
                cascade_frame = pd.concat(
                    [data[data["uri"] == uri], data[data["subject_uri"] == uri]]
                )

                cascade_frame = cascade_frame.sort_values(
                    "clean_dt", ascending=True
                ).reset_index(drop=True)

                # This happens for 19 cascades.
                if cascade_frame.iloc[0]["type"] != "post":
                    print("\t - Skip cascade: original post comes after a repost.")
                    continue

                # Past this point we create and store cascades, so we can now store the uri
                all_uris.add(uri)
                os.makedirs(cascade_dir, exist_ok=True)

                num_posts = len(cascade_frame)
                print(f"\t - Number of posts: {num_posts:,}")

                # We only need to generate one version of cascades len 2 because we are not guessing.
                num_cascade_versions = 1 if (num_posts == 2) else N_SIMULATIONS

                # Create lists for the function below
                poten_edge_users = list(cascade_frame["author"])
                poten_edge_tstamps = [
                    int(dt.timestamp()) for dt in cascade_frame.clean_dt
                ]
                poten_edge_fcounts = np.array(cascade_frame["followers_count"])

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
                    g.cascade_id = uri
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

    output_uri_filepath = os.path.join(DATA_DIR, "final_uris_list.txt")
    with open(output_uri_filepath, "w") as file:
        # Write each string from the set to the file on a new line
        for uri in all_uris:
            file.write(uri + "\n")

    print("#" * 50)
    print(f"Script complete!")
