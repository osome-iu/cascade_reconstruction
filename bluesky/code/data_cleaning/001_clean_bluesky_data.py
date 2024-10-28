"""
Purpose: Clean BlueSky data collected with the Firehose streamer.
    - Ref: https://atproto.blue/en/latest/atproto_firehose/index.html

Input:
    - None
    - Path to the BlueSky data collected with the Firehose streamer set with a constant below.

Output:
    - .parquet file with cleaned data

Author:
    - Matthew DeVerna
"""

import gzip
import json
import os

import pandas as pd


# Ensure we are in the scripts directory for relative paths
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

DATA_DIR = "/data_volume/cascade_reconstruction/bluesky/firehose_data"
OUPUT_DIR = "/data_volume/cascade_reconstruction/bluesky/clean_firehose_data"
os.makedirs(OUPUT_DIR, exist_ok=True)

COLUMNS2KEEP = [
    "type",
    "uri",
    "author",
    "cid",
    "createdAt",
    "text",
    "langs",
    "url",
    "subject_uri",
    "subject_cid",
]


def extract_file_date(path):
    # Example path:
    #  -> /data_volume/cascade_reconstruction/bluesky/firehose_data/2024-03-01.json.gz
    return path.split("/")[-1].split(".")[0]


def lang_to_string(lang):
    """
    Convert language values to string. Language codes will be separated by "|".

    Parameters:
    -----------
    - lang (nan [reposts], list): from the "langs" field in BlueSky data.

    Returns:
    -----------
    - string if lang is list
    - as is if nan
    """
    try:
        if not isinstance(lang, list):
            return lang
        elif len(lang) == 1:
            return lang[0]
        else:
            return "|".join(lang)
    except:
        print(lang)


def uri_to_bsky_url(uri):
    """
    Convert a Bluesky URI string to a corresponding webpage URL.

    Parameters:
    - uri (str): The URI string.

    Returns:
    - str: The corresponding webpage URL.
    """
    if "app.bsky.feed.post" not in uri:
        raise Exception(
            "URLs can only be generated for the action type 'app.bsky.feed.post'"
        )

    # Base URL for the transformation
    base_url = "https://bsky.app/profile"

    # Extract the URI identifier and the post ID
    # Example: 'at://did:plc:2vw2ctgmh5vzfjnr72ezcktm/app.bsky.feed.like/3kod4jlztdb2p'
    parts = uri.split("/")
    did_identifier = parts[2]  # These are unique identifiers for users
    post_id = parts[4]  # Extracts post ID, e.g., '3k57juz37hs2e'

    # Construct the URL
    url = f"{base_url}/{did_identifier}/post/{post_id}"

    return url


def clean_post(post_object):
    """
    Return a cleaned post object. Repost subjects (the post reposted), will
    be unnested. Post URLs will be generated from the URI. URIs only point to
    the original post. I.e., repost URLs will point to the reposted post.

    Parameters:
    -----------
    - post_object (dict): The post object from the BlueSky firehose.

    Returns:
    -----------
    - dict: The cleaned post object.
    """

    if "subject" in post_object:
        post_object["subject_uri"] = post_object["subject"].get("uri")
        post_object["subject_cid"] = post_object["subject"].get("cid")

    if "subject" in post_object:
        post_object["url"] = uri_to_bsky_url(post_object["subject_uri"])
        del post_object["subject"]
        return post_object
    else:
        post_object["url"] = uri_to_bsky_url(post_object["uri"])
    return post_object


files = [os.path.join(DATA_DIR, file) for file in sorted(os.listdir(DATA_DIR))]


for file_path in files:

    # Skip completed files
    file_date = extract_file_date(file_path)
    output_path = os.path.join(OUPUT_DIR, f"{file_date}__clean_bsky.parquet")
    if os.path.exists(output_path):
        print(f"Skipping {file_date}")
        continue

    print(f"Processing {file_date}")
    records = []
    with gzip.open(file_path, "rb") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)

            # Filter out deletions, irrelevant creations
            # (likes, blocks, follows, lists, etcs.)
            # and non-english posts
            # --- --- --- --- --- --- --- --- ---
            is_creation = record["action"] == "create"

            # Covers following types: 'app.bsky.feed.post' and 'app.bsky.feed.repost'
            rtype = record["type"]
            is_post = ("app.bsky" in record["type"]) and ("post" in rtype)

            # This may be incorrect as posters can set a list of w/e languages they want
            is_english = record["type"] == ["en"]

            if (not is_creation) or (not is_post):
                continue

            # Exclude quotes and replies
            # https://docs.bsky.app/docs/advanced-guides/posts#replies-quote-posts-and-embeds
            # --- --- --- --- --- --- --- --- ---
            # Note: 'parent' here is an old schema but there are a few edge cases that still include it (< 5)
            is_reply = ("reply" in record) or ("parent" in record)
            embed_type = record.get("embed", {}).get("$type")
            is_quote = (embed_type is not None) and (
                embed_type == "app.bsky.embed.record"
            )
            if is_reply or is_quote:
                continue

            try:
                records.append(clean_post(record))
            except:
                print(record)
                print("-" * 50)

    # Convert to dataframe and clean
    df = pd.DataFrame.from_records(records)
    df = df[COLUMNS2KEEP]
    df.type = df.type.map(
        {"app.bsky.feed.post": "post", "app.bsky.feed.repost": "repost"}
    )
    df.langs = df.langs.map(lang_to_string)

    # Save to disk
    df.to_parquet(output_path, index=False)
    del df  # Release memory

    print("#" * 50)
    print("#" * 50)
    print(f"Completed {file_date}")
    print("#" * 50)
    print("#" * 50)
