"""
Purpose: Collect the profile information for all authors in our sampled cascades.

Input:
    - None
    - Path to the sampled BlueSky cascades data

Output:
    - .parquet file with downloaded actor data

Author:
    - Matthew DeVerna
"""

import os
import random
import time

import pandas as pd

from atproto import Client

BSKY_DATA_DIR = "/data_volume/cascade_reconstruction/bluesky"
CASCADES_FILE = "sampled_cascades/sampled_cascades.parquet"
OUTPUT_DIR = "author_profiles"


def chunk_list(lst, n):
    """
    Chunk a list into smaller chunks of size n. Last list may be smaller than n.
    """
    return [lst[i : i + n] for i in range(0, len(lst), n)]


if __name__ == "__main__":

    cascades_file = os.path.join(BSKY_DATA_DIR, CASCADES_FILE)
    output_path = os.path.join(BSKY_DATA_DIR, OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)

    cascades_data = pd.read_parquet(cascades_file)
    authors = set(cascades_data["author"])

    bsky_password = os.environ.get("BSKY_PASSWORD")
    print(bsky_password)
    client = Client()
    client.login("matthewdeverna.com", bsky_password)

    # 25 is the maximum size number of profiles that can be requested per API call
    sorted_authors = sorted(authors)
    list_of_author_lists = chunk_list(sorted_authors, 25)

    num_author_lists = len(list_of_author_lists)

    for idx, authors in enumerate(list_of_author_lists):
        prop_done = (idx + 1) / num_author_lists
        print(f"Working on list {idx+1}/{num_author_lists} ({prop_done:.2%})")

        # Pre-sorting the list above ensures the authors in each list are the same
        # each time we run the script.
        full_output_path = os.path.join(output_path, f"{idx}_authors.parquet")
        if os.path.exists(full_output_path):
            print("Skipping already collected authors...")
            continue

        try:
            # Ref: https://docs.bsky.app/docs/api/app-bsky-actor-get-profiles
            result = client.get_profiles(authors)

            records = []
            for profile in result.profiles:

                records.append(
                    {
                        "did": profile.did,
                        "handle": profile.handle,
                        "description": profile.description,
                        "followers_count": profile.followers_count,
                        "follows_count": profile.follows_count,
                        "description": profile.description,
                        "posts_count": profile.posts_count,
                        "indexed_at": profile.indexed_at,
                    }
                )

            author_df = pd.DataFrame.from_records(records)
            author_df.to_parquet(full_output_path)

        except Exception as e:
            print(f"Exception on author list {idx}.\n\nError: {e}")

        finally:
            wait_time = random.randint(1, 3)
            print(f"Sleeping {round(wait_time,2)} seconds...")
            time.sleep(wait_time)

    print("--- Script complete ---")
