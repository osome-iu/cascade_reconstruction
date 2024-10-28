"""
Purpose:
    Automatically extract one or all entities (text, url, hashtag) from daily Twitter streaming data

    NOTE:
        1. All original (base-level) tweet entities are added
        2. Entities from `retweeted_status` objects are never added because they will be captured
            already in their original form
        3. Entities from `quoted_status` objects are ALWAYS added because we treat them as additional
            exposures of the embedded entities

Inputs:
    .json.gzip files in data dir specified by constants (see below)

Outputs:
    Four .parquet files (one for each entity) in intermediate dir specified by config file
    the common fields are:
    - post_id (str): id_str of the tweet
    - user_id (str): id_str of user
    - timestamp (int): UNIX timestamp of post (always represents the time the base post was sent)
    - tweet_type (str): options -> {original, retweet, quote}
    - retweeted_user_id (str, None): id_str of the retweeted poster. Only filled if the base tweet
        is a retweet, otherwise None
    - retweeted_post_id (str, None): id_str of the retweeted tweet. Only filled if the base tweet
        is a retweet, otherwise None
    - quoted_user_id (str, None): id_str of a quoted user. Only filled if the base tweet is a
        quote, otherwise None
    - quoted_post_id (str, None): id_str of a quoted user. Only filled if the base tweet is a
        quote, otherwise None
    - from_quoted (bool): whether or not the entity was taken from an embedded quoted_status object.
        True = entities taken from the quoted status object
        False = entities from taken from the base tweet object

    Entity specific fields in addition to those above for:
    - text file:
        - text (str): full tweet text
    - url file:
        - raw_url (str): raw url
    - hashtag file:
        - hashtag (str): hashtag

Authors:
   Matthew DeVerna & Bao Tran Truong
"""

import datetime
import glob
import gzip
import json
import os

import pandas as pd

from copy import deepcopy
from midterm.utils import get_logger
from midterm.data_model import Tweet

SCRIPT_PURPOSE = "This script extracts text, URLs, media URLs, and hashtags from daily Twitter streaming data"

LOG_DIR = "/data_volume/midterm_data/logs"
MIDTERM_DIR = "/data_volume/midterm_data"
DATA_DIR = os.path.join(MIDTERM_DIR, "streaming_data")
FMATCH_STRING = "streaming_data*.json.gz"
ENTITIES = ["text", "url", "hashtag"]

ENTITIES_DIR_MAP = {
    "text": os.path.join(MIDTERM_DIR, "entities", "text"),
    "urls": os.path.join(MIDTERM_DIR, "entities", "urls"),
    "hashtags": os.path.join(MIDTERM_DIR, "entities", "hashtags"),
}


def extract_entities(tweets_path):
    """
    Extract entities from raw Twitter data.
    Return four record structures for the major data entities: text, url, media, hashtags

    Parameters:
    -----------
    - tweets_path (str): full path to .json.gz file of tweet

    Returns:
    -----------
    - text_data (list of dict): URLs
    - url_data (list of dict): text
    - hashtag_data (list of dict): hashtags
    """

    # For output data
    text_data = []
    hashtag_data = []
    url_data = []

    # Script management variables
    num_skipped_posts = 0

    try:
        with gzip.open(tweets_path, "rb") as f:
            for line in f:
                try:
                    tweet = Tweet(json.loads(line.decode("utf-8")))

                    if not tweet.is_valid():
                        logger.info("SKIPPING NON-VALID TWEET")
                        logger.info(tweet.post_object)
                        continue

                    # Data dict shared only for different base-level entities
                    base_tweet_id = tweet.get_post_ID()
                    base_user_id = tweet.get_user_ID()
                    base_timestamp = tweet.get_timestamp()
                    base_info = {
                        "post_id": base_tweet_id,
                        "user_id": base_user_id,
                        "timestamp": base_timestamp,
                        "tweet_type": "original",
                        "retweeted_user_id": None,
                        "retweeted_post_id": None,
                        "quoted_user_id": None,
                        "quoted_post_id": None,
                        "from_quoted_status": False,
                    }
                    quoted_post_id = None

                    # If both retweet and quote status objects are present, this is a RT of a quoted status.
                    # This means we do not want to keep the RTd status entities, but we do want to mark that it is
                    # a retweet and from whom. Also, we keep the quoted_status object, handled below.
                    if all(flag for flag in [tweet.is_retweet, tweet.is_quote]):
                        base_info["tweet_type"] = "retweet"
                        base_info["retweeted_user_id"] = tweet.get_retweeted_user_ID()
                        base_info["retweeted_post_id"] = tweet.get_retweeted_post_ID()

                        quoted_post_id = tweet.get_value(["quoted_status", "id_str"])

                    # If we find a regular retweet, we do not want to keep those entities, but we
                    # should mark where it came from
                    elif tweet.is_retweet and (not tweet.is_quote):
                        base_info["tweet_type"] = "retweet"
                        base_info["retweeted_user_id"] = tweet.get_value(
                            ["retweeted_status", "user", "id_str"]
                        )
                        base_info["retweeted_post_id"] = tweet.get_value(
                            ["retweeted_status", "id_str"]
                        )

                    # If we find a quoted status, we want to mark that for the base tweet IDs
                    elif (not tweet.is_retweet) and tweet.is_quote:
                        base_info["tweet_type"] = "quote"
                        base_info["quoted_user_id"] = tweet.get_value(
                            ["quoted_status", "user", "id_str"]
                        )

                        quoted_post_id = tweet.get_value(["quoted_status", "id_str"])
                        base_info["quoted_post_id"] = quoted_post_id

                    ## Adding base tweet entities ##
                    ## -------------------------- ##
                    base_text_info = deepcopy(base_info)
                    base_text_info["text"] = tweet.get_text()
                    text_data.append(base_text_info)

                    base_hashtags = tweet.get_hashtags()
                    if len(base_hashtags) > 0:
                        for tag in base_hashtags:
                            base_tag_info = deepcopy(base_info)
                            base_tag_info["hashtag"] = tag
                            hashtag_data.append(base_tag_info)

                    base_urls = tweet.get_urls()
                    if len(base_urls) > 0:
                        for url in base_urls:
                            base_url_info = deepcopy(base_info)
                            base_url_info["raw_url"] = url
                            url_data.append(base_url_info)

                    ## Adding quoted tweet entities ##
                    ## ---------------------------- ##
                    if quoted_post_id is not None:

                        quoted_tweet = Tweet(tweet.get_value(["quoted_status"]))
                        quote_info = {
                            "post_id": base_tweet_id,
                            "user_id": base_user_id,
                            "timestamp": base_timestamp,
                            "tweet_type": "quote",
                            "retweeted_user_id": None,
                            "retweeted_post_id": None,
                            "quoted_user_id": quoted_tweet.get_user_ID(),
                            "quoted_post_id": quoted_tweet.get_post_ID(),
                            "from_quoted_status": True,
                        }

                        quote_text_info = deepcopy(quote_info)
                        quote_text_info["text"] = quoted_tweet.get_text()
                        text_data.append(quote_text_info)

                        quote_hashtags = quoted_tweet.get_hashtags()
                        if len(quote_hashtags) > 0:
                            for tag in quote_hashtags:
                                quote_tag_info = deepcopy(quote_info)
                                quote_tag_info["hashtag"] = tag
                                hashtag_data.append(quote_tag_info)

                        quote_urls = quoted_tweet.get_urls()
                        if len(quote_urls) > 0:
                            for url in quote_urls:
                                quote_url_info = deepcopy(quote_info)
                                quote_url_info["raw_url"] = url
                                url_data.append(quote_url_info)

                except Exception as e:
                    logger.exception("Error parsing a tweet")
                    logger.info(e)
                    logger.error(tweet.post_object)
                    num_skipped_posts += 1
                    continue

        logger.info(f" - Num. of weird skipped posts: {num_skipped_posts}")
        return text_data, hashtag_data, url_data

    except EOFError as e:
        logger.info(f" - Handling bad ending of a file...")
        return text_data, hashtag_data, url_data


if __name__ == "__main__":
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    logger = get_logger(
        log_dir=LOG_DIR,
        log_fname=f"extract_entities--{today}.log",
        also_print=True,
    )

    input_files = sorted(glob.glob(os.path.join(DATA_DIR, FMATCH_STRING)))

    logger.info("#" * 10)
    logger.info(f" - Extracting from files found here: {DATA_DIR}")
    logger.info(f" - Files to process: {len(input_files)}")
    logger.info(f" - Entities to extract: {ENTITIES}")
    logger.info("#" * 10)

    logger.info(" - Mapping dates in range to input files...")
    for file in input_files:

        # Example filename: streaming_data--2022-12-30.json.gz
        file_basename = os.path.basename(file)
        file_date_str = file_basename.split("--")[-1].replace(".json.gz", "")

        logger.info("-" * 10)
        logger.info(f" - Working on {file_basename}")

        try:
            # Extract data
            text_data, hashtag_data, url_data = extract_entities(file)
            entity_data_dict = {
                "text": text_data,
                "hashtags": hashtag_data,
                "urls": url_data,
            }

            # Save data
            for entity, records in entity_data_dict.items():
                outdir = ENTITIES_DIR_MAP[entity]
                os.makedirs(outdir, exist_ok=True)
                outpath = os.path.join(
                    outdir,
                    f"{entity}--{file_date_str}.parquet",
                )
                entity_df = pd.DataFrame.from_records(records)
                entity_df.drop_duplicates(inplace=True)
                entity_df.to_parquet(outpath, index=None, engine="pyarrow")

        except Exception as e:
            logger.error(f"Error extracting entities for the date {file_date_str}: {e}")
            continue

    logger.info("Script complete")
    logger.info("-" * 50)
