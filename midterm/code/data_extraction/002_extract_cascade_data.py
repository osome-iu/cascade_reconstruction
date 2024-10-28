"""
Purpose: Extract data for sampled cascades from raw tweet data.

Input: None
    - Reads data files based on hardcoded paths (see constants below).

Output: Pandas dataframe containing the columns listed below.
    - Filename: `cascades_dataframe.parquet`
        - Each line represents an individual tweet
    - Columns:
        - 

Author: Matthew DeVerna
"""

import gzip
import json
import os

import pandas as pd

# Local package
from midterm.utils import get_files_in_date_range
from midterm.data_model import Tweet

# Ensure current directory is the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SAMPLED_CASCADES_PATH = "../../data/intermediate_files/sampled_cascades.txt"
DATA_DIR = "/data_volume/midterm_data/streaming_data"
OUTPUT_DIR = "../../data/sampled_cascades_records/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date range for files read
FILES_START_DATE = "2022-11-02"
FILES_END_DATE = "2022-11-15"


def load_sampled_cascade_ids(path):
    """
    Load sampled cascade IDs.

    Note:
    ----------
    - File contains a single twitter ID per line

    Parameters:
    ----------
    - path (str) : path to file

    Returns:
    ----------
    - sampled_cascade_ids (set) : list of sampled cascade IDs
    """
    with open(path, "r") as f:
        sampled_cascade_ids = [line.strip() for line in f.readlines()]
    return set(sampled_cascade_ids)


def extract_cascade_data(filtered_files, sampled_cascade_ids):
    """
    Extract data for sampled cascades from raw tweet data.

    Parameters:
    ----------
    - filtered_files (list) : list of filtered files
    - sampled_cascade_ids (set) : set of sampled cascade IDs

    Returns:
    ----------
    - df (pandas.DataFrame) : dataframe containing data for sampled cascades
    """

    records = []
    seen_cascades = set()

    for file in filtered_files:
        print(f"\t - Loading {file}...")
        with gzip.open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                tweet_obj = Tweet(data)

                if not tweet_obj.is_retweet:
                    tweet_id = tweet_obj.get_post_ID()

                    # In this case their is no RT object and the original tweet
                    # is not in our sampled cascades, so we skip.
                    if tweet_id not in sampled_cascade_ids:
                        continue

                    # If this original tweet is somehow already in the data,
                    # it means we can skip it
                    if tweet_id in seen_cascades:
                        continue

                    # If we get this far, we have a tweet that is in our sampled
                    # of cascades and must add the record
                    records.append(
                        {
                            "cascade_id": tweet_id,
                            "tweet_id": tweet_id,
                            "is_root": True,
                            "user_id": tweet_obj.get_user_ID(),
                            "timestamp": tweet_obj.get_timestamp(),
                            "follower_count": tweet_obj.get_follower_count(),
                            "text": tweet_obj.get_text(),
                            "created_at": tweet_obj.get_created_at(),
                        }
                    )

                    # Mark that this cascade ID has been seen
                    seen_cascades.add(tweet_id)

                else:
                    # Here, we have a retweet, so we need to check if the retweet
                    # status is in our sampled cascades. If not, we can skip.
                    rt_id = tweet_obj.retweet_object.get_post_ID()
                    if rt_id not in sampled_cascade_ids:
                        continue

                    # If we get this far, we have a retweet of one of our sampled
                    # cascades so we must add the original tweet as well as the
                    # retweeted status (but only if we have not seen it yet).

                    # First, we add the top-level tweet
                    records.append(
                        {
                            "cascade_id": rt_id,  # Must point to the original
                            "tweet_id": tweet_id,
                            "is_root": False,
                            "user_id": tweet_obj.get_user_ID(),
                            "timestamp": tweet_obj.get_timestamp(),
                            "follower_count": tweet_obj.get_follower_count(),
                            "text": tweet_obj.get_text(),
                            "created_at": tweet_obj.get_created_at(),
                        }
                    )

                    # Second, check if we have already seen the retweeted tweet
                    rt_id = tweet_obj.retweet_object.get_post_ID()
                    if rt_id in seen_cascades:
                        continue

                    # If not, we add the retweeted tweet
                    records.append(
                        {
                            "cascade_id": rt_id,
                            "tweet_id": rt_id,
                            "is_root": True,
                            "user_id": tweet_obj.retweet_object.get_user_ID(),
                            "timestamp": tweet_obj.retweet_object.get_timestamp(),
                            "follower_count": tweet_obj.retweet_object.get_follower_count(),
                            "text": tweet_obj.retweet_object.get_text(),
                            "created_at": tweet_obj.retweet_object.get_created_at(),
                        }
                    )

                    seen_cascades.add(rt_id)

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    print("Selecting correct files...")
    print(f"\t - Start date: {FILES_START_DATE}")
    print(f"\t - End date: {FILES_END_DATE}\n")

    print("Loading sampled cascade IDs...")
    sampled_cascade_ids = load_sampled_cascade_ids(SAMPLED_CASCADES_PATH)

    print("\nExtracting cascade data...")
    filtered_files = get_files_in_date_range(FILES_START_DATE, FILES_END_DATE, DATA_DIR)
    records_df = extract_cascade_data(filtered_files, sampled_cascade_ids)

    print("\nSummary:")
    print(f"\t - Number of cascades found: {records_df.cascade_id.nunique():,}")
    print(f"\t - Number of records found: {len(records_df):,}")

    num_duplicates = records_df.duplicated().sum()
    print(f"\t - Number of duplicates found: {num_duplicates:,}\n")

    if num_duplicates > 0:
        print("Dropping duplicates...")
        records_df = records_df.drop_duplicates().reset_index(drop=True)
        print("SUMMARY:")
        print(f"\t - New number of records found: {len(records_df):,}\n")

    # We noticed one account with a follower count of -1
    # We correct negative values here by setting their follower count to 0
    negative_records = records_df[records_df.follower_count < 0]
    num_negative_records = len(negative_records)
    if num_negative_records > 0:
        print(f"Correcting {num_negative_records} negative follower counts...")
        records_df.loc[negative_records.index, "follower_count"] = 0

    print("Saving...")
    print(f"\t - Output directory: {OUTPUT_DIR}")
    records_df.to_parquet(os.path.join(OUTPUT_DIR, "cascade_records.parquet"))
