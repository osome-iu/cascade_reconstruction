"""
Purpose: Sample cascades from the cleaned BlueSky data.

Input:
    - None
    - Path to the cleaned BlueSky .parquet file

Output:
    - .parquet file with sampled cascade data

Author:
    - Matthew DeVerna
"""

import os
import random

import pandas as pd

from datetime import datetime, timedelta

DATA_DIR = "/data_volume/cascade_reconstruction/bluesky/clean_firehose_data"
OUPUT_DIR = "/data_volume/cascade_reconstruction/bluesky/sampled_cascades"
os.makedirs(OUPUT_DIR, exist_ok=True)

NUM_POSTS = 5_000


def sample_posts(input_dir, output_dir, num_posts, random_state=42):
    """
    Sample posts from cleaned BlueSky data.

    Parameters:
    -----------
    - input_dir (str): Path to the cleaned BlueSky .parquet file
    - output_dir (str): Output directory for the sampled posts
    - num_posts (int): Number of posts to sample
    - random_state (int): Random state for sampling

    Returns:
    - None
    """

    print(
        f"Sampling {num_posts} posts from " f"{input_dir} and saving to {output_dir}..."
    )

    # Select the first week to sample from
    files = [os.path.join(input_dir, file) for file in sorted(os.listdir(input_dir))]
    files = files[:14]

    dfs = []
    for file in files:
        print(f"Loading {os.path.basename(file)}...")
        df = pd.read_parquet(file)
        dfs.append(df)

    all_dfs = pd.concat(dfs).reset_index(drop=True)

    # Users can input ANY time string they want. This ensures we narrow to those in the right month.
    right_month_mask = all_dfs["createdAt"].str.startswith("2024-03")
    num_non_right_month = sum(~right_month_mask)
    all_dfs = all_dfs[right_month_mask].reset_index(drop=True)

    print(f"Number of posts from the wrong month: {num_non_right_month:,}")

    # There are many weird formatted times that do not match the standard atproto protocol.
    # They all BEGIN with the same 19 characters, what changes after that is the level of detail
    # with respect to timezone, milliseconds, etc. — to solve I simply cut them off.
    all_dfs["clean_dt"] = pd.to_datetime(all_dfs["createdAt"].str[:19])

    # Still some
    start = datetime.strptime("2024-03-01", "%Y-%m-%d")
    end = datetime.strptime("2024-03-15", "%Y-%m-%d")

    right_time_mask = (all_dfs["clean_dt"] >= start) & (all_dfs["clean_dt"] < end)
    num_non_right_time = sum(~right_time_mask)
    all_dfs = all_dfs[right_time_mask].reset_index(drop=True)

    print(f"Number of posts from the wrong date range: {num_non_right_time:,}")

    # Get posts
    posts = all_dfs[all_dfs["type"] == "post"].reset_index(drop=True)

    # Use their URIs to extract reposts
    post_uris = set(posts["uri"])
    reposts = all_dfs[all_dfs["subject_uri"].isin(post_uris)].reset_index(drop=True)

    # Post URIs with at least one repost (value_counts() by default gets rid of None values)
    post_uris_w_repost = all_dfs.subject_uri.value_counts().index.to_list()

    # Find posts from the first seven days based on these URIs
    first_week_posts = posts[
        (posts["clean_dt"] < start + timedelta(days=8))
        & (posts["uri"].isin(post_uris_w_repost))
    ].reset_index(drop=True)

    # Create a set of their URIs
    first_week_uris = set(first_week_posts["uri"])
    print(f"Sample 5,000 cascades from {len(first_week_uris):,} original posts.")

    # Sample 5000 posts
    random.seed(3)
    sampled_uris = random.sample(list(first_week_uris), k=NUM_POSTS)

    # Select sampled posts and reposts
    sampled_posts = posts[posts["uri"].isin(sampled_uris)].reset_index(drop=True)
    sampled_reposts = reposts[reposts["subject_uri"].isin(sampled_uris)]

    err_msg = "Wrong number of items!"
    assert sampled_posts["uri"].nunique() == NUM_POSTS, "POSTS: " + err_msg
    assert sampled_reposts["subject_uri"].nunique() == NUM_POSTS, "REPOSTS: " + err_msg

    # Combine the two and save as .parquet
    sampled_cascades = pd.concat([sampled_posts, sampled_reposts]).reset_index(
        drop=True
    )
    sampled_cascades.to_parquet(os.path.join(OUPUT_DIR, "sampled_cascades.parquet"))


if __name__ == "__main__":
    sample_posts(
        input_dir=DATA_DIR,
        output_dir=OUPUT_DIR,
        num_posts=NUM_POSTS,
    )

    print("--- Script complete ---")
