"""
Purpose: Clean the `chunked_raw_data_anon` data so it is ready for analysis.

How the data is altered:
- Remove all cascades for which we are missing any of the below data for any tweets in the cascade:
    - `user_followers`: number of followers the poster has
    - `veracity`: whether the or not the cascade is 'true', 'false', or 'mixed'
- Clean up the veracity values that we do have and map them to: 'true', 'false', or 'mixed'
- Convert id columns to strings
- Convert `user_account_age` column to int
- Convert `tweet_date` column to pandas.datetime objects

Input:
- We read the raw chunked data from: `chunked_raw_data_anon`

Output:
- .parquet file

Author: Matthew DeVerna
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from copy import deepcopy

# The directory where the raw data is stored
DATA_DIR = "../../vosoughi_replication_code/data/chunked_raw_data_anon/"
OUT_DIR = "../../cleaned_data/"
VERACITY_MAP = {
    False: "false",
    "FALSE": "false",
    "False": "false",
    True: "true",
    "TRUE": "true",
    "MIXED": "mixed",
    "r_33": None,
    "r_45": None,
}


# Create an ArgumentParser object
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean the data based on the provided arguments."
    )
    parser.add_argument(
        "--keep-all",
        "-k",
        action="store_true",
        help="When included, keep all cascades possible. Exclude for full cleaning",
    )
    return parser.parse_args()


def load_raw_data(dir, matching_string):
    """
    Load the raw data.
    """
    files = sorted(glob.glob(os.path.join(dir, matching_string)))
    # Other columns cannot be set as they contain multiple types of values
    dtypes = {
        "tid": str,
        "cascade_id": str,
        "parent_tid": str,
        "user_engagement": float,
        "cascade_root_tid": str,
    }
    data = []
    for file in files:
        print("-" * 50)
        print(f"Loading {os.path.basename(file)}...")
        data.append(pd.read_csv(file, dtype=dtypes))
        print("-" * 50)
    return pd.concat(data).reset_index(drop=True)


def map_none(val):
    """
    Maps the string value "None" to the None object.

    Parameters:
    -----------
    - val (str): The string value to be mapped.

    Returns:
    -----------
    - object: The mapped value. If the input value is "None", returns None.
        Otherwise, returns the input value as is.
    """
    if val == "None":
        return None
    else:
        return val


if __name__ == "__main__":
    args = parse_args()
    keep_all = args.keep_all

    # Load the raw data
    print("Loading raw data...".upper())
    data = load_raw_data(DATA_DIR, "*.csv")
    print(f"\t- Done.")

    # Get all columns that contain 'none' strings
    columns_w_none = []
    print("columns with 'none' strings:".upper())
    for col in data.columns:
        if data[col].isin(["None", "none"]).sum() > 0:
            print(f"\t-> {col}")
            columns_w_none.append(col)
    print(f"\t- Done.")

    if not keep_all:
        # Ensuring that columns with any 'none' have 'none' strings for all cells
        print("Checking 'None' values are consistent for all rows...".upper())
        none_str_indices = []
        print(f"Num rows in data: {data.shape[0]:,}")
        for idx, rowdata in data.iterrows():
            items = [
                rowdata.user_account_age,
                rowdata.user_verified,
                rowdata.user_followers,
                rowdata.user_followees,
            ]
            if idx % 500_000 == 0:
                print(f"Rows processed: {idx:,}")
            if any(x == "None" for x in items):
                if not all(x == "None" for x in items):
                    none_str_indices.append(idx)
        print(
            f"Number of rows where all values DO NOT == 'None': {len(none_str_indices)}"
        )
        print(f"\t- Done.")

    # Clean the `veracity` values
    print("Cleaning the `veracity` column...".upper())
    data.veracity = data.veracity.map(VERACITY_MAP)
    print(f"\t- Done.")

    print("Convert string times to datetimes".upper())
    data.tweet_date = pd.to_datetime(data.tweet_date)
    print(f"\t- Done.")

    columns_w_none = [
        "user_account_age",
        "user_verified",
        "user_followers",
        "user_followees",
    ]
    if keep_all:
        print("Keeping all cascades possible.".upper())
        print("Dropping columns with 'none' values.".upper())

        fname = "clean_raw_data_anon_ALL.parquet"

        # Here we only check for NaN values in the veracity column. All else should be okay...
        cascades_w_missing_info = set(data[data.veracity.isna()]["cascade_id"])
        missing_data_df = data[data.cascade_id.isin(cascades_w_missing_info)].copy()

    else:
        fname = "clean_raw_data_anon.parquet"
        # In theory all of these columns below are not important, however, the cell above
        # shows that if one is None, all are None.
        print("Converting string 'None' values to real Pyhton None values...".upper())
        for col in columns_w_none:
            data[col] = data[col].apply(map_none)
        print(f"\t- Done.")

        cascades_w_missing_info = set(data[data.isna().any(axis=1)]["cascade_id"])
        missing_data_df = data[data.cascade_id.isin(cascades_w_missing_info)].copy()

    print("Calculating stats on cascades with missing data...".upper())
    total_cascades = data["cascade_id"].nunique()
    total_cascades_w_missing_info = len(cascades_w_missing_info)
    total_cascades_wo_missing_info = total_cascades - total_cascades_w_missing_info
    prop_cascades_w_missing_info = total_cascades_w_missing_info / total_cascades

    print(f"\t- Total cascades: {total_cascades:,}")
    print(f"\t- Total w/ missing info.: {total_cascades_w_missing_info:,}")
    print(f"\t- Total w/o missing info.: {total_cascades_wo_missing_info:,}")
    print(f"\t- Percent of cascades we drop = {prop_cascades_w_missing_info:%}")
    print("-" * 50)

    missing_data_boolean = data.cascade_id.isin(cascades_w_missing_info)
    num_tweets_to_removed = len(data[missing_data_boolean])
    total_retweets = len(data)
    prop_retweets_to_remove = missing_data_boolean.sum() / total_retweets

    print(f"\t- Total retweets: {total_retweets:,}")
    print(f"\t- Total retweets being dropped: {missing_data_boolean.sum():,}")
    print(f"\t- Percentage of retweets being dropped: {prop_retweets_to_remove:%}")

    print("Dropping rows with missing data...".upper())
    data = deepcopy(data[~missing_data_boolean].reset_index(drop=True))

    print(f"\t- Total retweets remaining: {len(data):,}")
    print("-" * 50)

    if not keep_all:
        print("Set columns to type numeric...".upper())
        columns_w_none.remove("user_verified")
        for col in columns_w_none:
            print(f"\t-> {col}")
            data[col] = data[col].map(int)
        print(f"\t- Done.")

    print("Convert `was_retweeted` to boolean...".upper())
    data.was_retweeted = data.was_retweeted.map(bool)
    print(f"\t- Done.")

    if not keep_all:
        print("Dropping cascades with length 1...".upper())
        num_cascades = data["cascade_id"].nunique()
        lengths = (
            data.groupby("cascade_id").size().to_frame("cascade_length").reset_index()
        )
        cascades = lengths[lengths.cascade_length > 1].cascade_id.to_list()
        data = data[data["cascade_id"].isin(cascades)]
        num_cascades_remaining = data["cascade_id"].nunique()
        num_cascades_removed = num_cascades - num_cascades_remaining
        print(f"\t- Total cascades: {num_cascades:,}")
        print(f"\t- Total cascades removed: {num_cascades_removed:,}")
        print(f"\t- Total cascades remaining: {num_cascades_remaining:,}")
        print(
            f"\t- Percent of cascades removed = {num_cascades_removed / num_cascades:%}"
        )
        print(f"\t- Done.")

    print("Saving data...".upper())
    if keep_all:
        # Split data into two frames and save each with prefix version
        for num, half_frame in enumerate(np.array_split(data, 2)):
            half_frame.to_parquet(
                os.path.join(OUT_DIR, f"v_{str(num).zfill(2)}_{fname}")
            )
    else:
        data.to_parquet(os.path.join(OUT_DIR, fname))
    print(f"\t- Done.")
    print("Script complete.")
