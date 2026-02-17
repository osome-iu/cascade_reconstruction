"""
Purpose:
    Estimate the power law alpha for each cascade.

Note:
    - Estimates are made for cascades of different minimum sizes.
    - Estimates for each minimum size is based on the mean estimate for a bootstrapped
    sample of cascades of that size.

Input:
    None. Paths are set and data is loaded in.

Output:
    .parquet file with the following columns:
        - run (int) : the run number
        - xmin (float) : the estimated xmin value (1 second)
        - alpha (float) : the estimated alpha value
        - min_rts (int) : the minimum number of retweets for that fitting simuliation
        - time_diff_type (str) : the type of time difference paradigm.
            - See `generate_time_differences_files.py` and generate_min_rts_dict() (below) for details.

Authors:
- Matthew DeVerna
"""
import json
import os

import numpy as np
import pandas as pd

from collections import defaultdict

from pkg.reconstruction import simulate_plaw_fits

# Directories
DATA_DIR = "../../output/time_differences"
OUTPUT_DIR = "../../output/power_law_alpha_estimations"

# Data file paths
STAR_TDIFF_PATH = os.path.join(DATA_DIR, "stars_time_differences.json")
MOST_RECENT_TDIFF_PATH = os.path.join(DATA_DIR, "most_recent_time_differences.json")

# Simulation parameters
XMIN = 1  # 1 = one second
SAMPLE_SIZE = 5_000  # with replacement
NUMBER_OF_FITS = 1_000


def generate_min_rts_dict(time_diff_dict, hard_split=False):
    """
    Split cascade time differences based on the minimum cascade size.

    Sizes: 10, 100, 1,000, 5,000, 10,000, 100,000

    Parameters:
    -----------
    - time_diff_dict (dict) : a dictionary of cascade time differences
        - key (int) : the cascade_id
        - value (dict) : a dictionary of time differences
            - keys: "length", "time_diffs"
            - values: int (cascade size), list of time differences (seconds)
    - hard_split (bool) : whether the time differences are separated by cutoffs
        - if True, the time differences in (e.g.) key 10 of the returned dictionary
            are only for cascades of length between 10 (inclusive) and 1,000 (exclusive).
        - if False, the time differences in (e.g.) key 10 of the returned dictionary
            are for cascades with length greater than or equal to 10.

    Return:
    ----------
    min_rts_dict (dict) : a dictionary containing time differences for given
        minimum cascade sizes
    """
    min_rts_dict = defaultdict(list)

    for cascade_data in time_diff_dict.values():
        cascade_size = cascade_data["length"]
        rt_time_diffs = cascade_data["time_diffs"]

        # All cascades
        min_rts_dict[-1].extend(rt_time_diffs)

        # Include RT time differences for only cascades BETWEEN the cutoffs below
        if hard_split:
            if cascade_size >= 100_000:
                min_rts_dict[100_000].extend(rt_time_diffs)
            elif cascade_size >= 10_000:
                min_rts_dict[10_000].extend(rt_time_diffs)
            elif cascade_size >= 5_000:
                min_rts_dict[5_000].extend(rt_time_diffs)
            elif cascade_size >= 1_000:
                min_rts_dict[1_000].extend(rt_time_diffs)
            elif cascade_size >= 100:
                min_rts_dict[100].extend(rt_time_diffs)
            elif cascade_size >= 10:
                min_rts_dict[10].extend(rt_time_diffs)
            elif cascade_size >= 1:
                min_rts_dict[1].extend(rt_time_diffs)
            else:
                raise ValueError("A strange value has been found.")

        # Include RT time differences for cascades AT LEAST AS LARGE as the cutoffs below
        else:
            if cascade_size >= 100_000:
                min_rts_dict[100_000].extend(rt_time_diffs)
            if cascade_size >= 10_000:
                min_rts_dict[10_000].extend(rt_time_diffs)
            if cascade_size >= 5_000:
                min_rts_dict[5_000].extend(rt_time_diffs)
            if cascade_size >= 1_000:
                min_rts_dict[1_000].extend(rt_time_diffs)
            if cascade_size >= 100:
                min_rts_dict[100].extend(rt_time_diffs)
            if cascade_size >= 10:
                min_rts_dict[10].extend(rt_time_diffs)
            if cascade_size >= 1:
                min_rts_dict[1].extend(rt_time_diffs)

    return min_rts_dict


if __name__ == "__main__":
    # Load data using the json module
    with open(STAR_TDIFF_PATH, "r") as f:
        star_tdiffs = json.load(f)
    with open(MOST_RECENT_TDIFF_PATH, "r") as f:
        most_recent_tdiffs = json.load(f)

    print("Splitting cascade time differences based on the minimum cascade size...")
    split_modes = ["hard_split", "equal_or_greater"]  # How to split cascades by size
    tdiff_dicts_list = []  # Will store all the dictionaries here
    for split_mode in split_modes:
        for tdiffs, label in [
            (star_tdiffs, "stars"),
            (most_recent_tdiffs, "most_recent"),
        ]:
            # Generate the dictionary using the appropriate split mode
            min_rts_dict = dict(
                generate_min_rts_dict(tdiffs, hard_split=(split_mode == "hard_split"))
            )
            tdiff_dicts_list.append((min_rts_dict, f"{label}_{split_mode}"))

    print("Convert min cascade size dictionary values to numpy arrays...")
    for tdiff_dict, label in tdiff_dicts_list:
        for min_rts, tdiffs in tdiff_dict.items():
            # Because we lose milliseconds, some time diffs may be zero, which breaks
            # the powerlaw function. We add a tenth of a second to all time diffs to fix this.
            tdiff_dict[min_rts] = np.array(tdiffs) + 0.01

    print("Estimating alpha for each cascade minimum length...")
    fit_frames = {}
    for tdiff_dict, label in tdiff_dicts_list:
        formatted_label = " ".join(label.split("_")).capitalize()
        print(f"\t--> Working on <{formatted_label}>...")
        for min_rts, tdiffs in tdiff_dict.items():
            print(f"\t\t--> Minimum cascade size: {min_rts}")
            results = simulate_plaw_fits(
                arr=tdiffs, sample_size=SAMPLE_SIZE, num_sims=NUMBER_OF_FITS, xmin=XMIN
            )
            fit_frames[(min_rts, label)] = results

    print("Combining results into a single data frame...")
    all_results = []
    for minRts_tdiffType, results in fit_frames.items():
        results["min_rts"] = minRts_tdiffType[0]
        results["time_diff_type"] = minRts_tdiffType[1]
        all_results.append(results)
    all_fits_frame = pd.concat(all_results)

    print("Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_fits_frame.to_parquet(
        os.path.join(OUTPUT_DIR, "power_law_alpha_estimations.parquet")
    )

    print("Script complete.")
