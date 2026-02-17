"""
Purpose:
    Create vectors of structure metrics for different versions of each cascade and
    calculate their mean cosine similarity for all parameter configurations.

Note:
    - Include the --tid flag to compare all PDI versions to the single TID version.
    - Exclude the --tid flag to compare different PDI methods to one another.

Input:
    None. Paths are set and data is loaded in.

Output:
    - .parquet file saved here: /output/cascade_structure_cosine_similarity/
    - Filename: structure_metrics_cosine_similarity_gamma_{gamma}_alpha_{alpha}_{cascade_type_string}.parquet"
        - ALPHA = alpha level
        - if TID: the PDI versions are compared only to the single TID version
        - if PDI: all PDI methods are compared to each other
    - Columns:
        - alpha : The value of the alpha parameter
        - gamma : The value of the gamma parameter
        - type (pdi or tid) : the which version cascades were compared to
        - cascade_id : The identifier for the cascade
        - mean_cosine_sim : The mean cosine similarity value
        - std_cosine_sim : The standard deviation of cosine similarity values

Authors:
- Matthew DeVerna
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from numpy.linalg import norm

FILES_DIR = "../../output/cascade_metrics"
OUT_DIR = "../../output/cascade_structure_cosine_similarity"
os.makedirs(OUT_DIR, exist_ok=True)

COLUMNS_OF_INTEREST = ["depth", "structural_virality", "max_breadth"]


def parse_args():
    description = (
        "Create vectors of structure metrics for different versions of each cascade"
        " and calculate their mean cosine similarity for all parameter configurations."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--type",
        required=True,
        choices=["midterm", "vosoughi"],
        help="Specify the cascade type to analyze.",
    )
    arg_msg = (
        "Include the --tid flag to compare all PDI versions to the single TID version. "
        "Exclude the --tid flag to compare different PDI methods to one another."
    )
    parser.add_argument(
        "--tid",
        action="store_true",
        help=arg_msg,
    )
    args = parser.parse_args()
    return args


def extract_parameters_from_filename(filename):
    """
    Extract parameter names and values from a given file name.

    Parameters:
    - filename (str): The file name from which to extract parameter names.

    Returns:
    - dict: A dictionary containing extracted parameter names and values.

    Example:
    >>> filename = "cascade_metrics_statistics_gamma_0_25_alpha_1_1.parquet"
    >>> extract_parameters_from_filename(filename)
    {'gamma': '0.25', 'alpha': '1.1'}
    """
    if not isinstance(filename, str):
        raise ValueError("filename must be a string.")

    # Split basename by underscores
    filename = os.path.basename(filename).replace(".parquet", "")
    parts = filename.split("_")

    # Find indices of parameter names
    gamma_index = parts.index("gamma")
    alpha_index = parts.index("alpha")

    # Parameter values
    gamma_value = float(".".join(parts[gamma_index + 1 : gamma_index + 3]))
    alpha_value = float(".".join(parts[alpha_index + 1 : alpha_index + 3]))

    # Return parameter dictionary
    return {"gamma": gamma_value, "alpha": alpha_value}


if __name__ == "__main__":
    args = parse_args()
    tid = args.tid
    cascade_type = f"{args.type}"
    if cascade_type == "midterm":
        raise Exception("Not implemented yet! Must update for new midterm data!")

    # Get data file paths
    pdi_files = sorted(
        glob.glob(os.path.join(FILES_DIR, "cascade_metrics_statistics*.parquet"))
    )
    tid_file = os.path.join(FILES_DIR, "time_inferred_diffusion_metrics.parquet")

    results_by_alpha = {}

    for file in pdi_files:
        param_dict = extract_parameters_from_filename(file)
        alpha = param_dict["alpha"]
        gamma = param_dict["gamma"]

        if tid:
            tid_df = pd.read_parquet(tid_file)
            tid_df = tid_df.set_index("cascade_id")  # Allows faster lookup (I think)

        cascade_type = "tid" if tid else "pdi"
        print("#" * 50)
        print(f"Cascade type: {cascade_type.upper()}")
        print(f"ALPHA: {alpha}")
        print(f"GAMMA: {gamma}")
        print("#" * 50)

        # Make file name
        gamma_str = str(gamma).replace(".", "_")
        alpha_str = str(alpha).replace(".", "_")
        fname_base = "structure_metrics_cosine_similarity_gamma"
        file_name = f"{fname_base}_{gamma_str}_alpha_{alpha_str}_{cascade_type}.parquet"
        output_file_path = os.path.join(
            OUT_DIR,
            file_name,
        )

        if os.path.exists(output_file_path):
            print("**Skipping because completed already.**")
            continue

        df = pd.read_parquet(file)

        cascade_ids = list(df.cascade_id.unique())
        num_cascades = len(cascade_ids)

        records = []
        for cas_num, cascade_id in enumerate(cascade_ids, start=1):
            if cas_num % 1_000 == 0:
                print(f"Working on cascade {cas_num}/{num_cascades} ...")

            metric_matrix = df.loc[
                df.cascade_id == cascade_id  # select the cascade of interest
            ][
                COLUMNS_OF_INTEREST  # select columns of interest
            ].to_numpy()  # convert to a matrix

            # If cascades size = 2, there will be only one version, so we skip
            if len(metric_matrix) == 1:
                continue

            # Normalize metric matrix from PDI
            norms = norm(metric_matrix, axis=1, keepdims=True)
            normalized_matrix = metric_matrix / norms

            if tid:
                # Compare `normalized_matrix` to the tid metrics
                tid_metric_vec = tid_df.loc[cascade_id][COLUMNS_OF_INTEREST].to_numpy()
                normalized_tid_vec = tid_metric_vec / norm(tid_metric_vec)

                similarity_values = list(
                    np.dot(normalized_matrix, normalized_tid_vec.T)
                )

            else:
                # Otherwise, we can just compare the PDI values to themselves
                cosine_similarity_matrix = np.dot(
                    normalized_matrix, normalized_matrix.T
                )

                # The above returns a matrix with duplicate similarities,
                # so we first select only the values we want from the upper triangle
                upper_triangle_indices = np.triu_indices_from(
                    cosine_similarity_matrix, k=1
                )
                similarity_values = list(
                    cosine_similarity_matrix[upper_triangle_indices]
                )

            # Calculate the mean and std of the resulting similarity values
            mean_similarity = np.mean(similarity_values)
            std_similarity = np.std(similarity_values)
            records.append(
                {
                    "alpha": alpha,
                    "gamma": gamma,
                    "type": cascade_type,
                    "cascade_id": cascade_id,
                    "mean_cosine_sim": mean_similarity,
                    "std_cosine_sim": std_similarity,
                }
            )

        # Convert to DF, add the size of the cascade and save
        sim_df = pd.DataFrame.from_records(records)
        results_by_alpha[alpha] = sim_df
        sims_with_size = pd.merge(
            df[["cascade_id", "size"]].drop_duplicates(), sim_df, on="cascade_id"
        )
        sims_with_size.to_parquet(output_file_path)
