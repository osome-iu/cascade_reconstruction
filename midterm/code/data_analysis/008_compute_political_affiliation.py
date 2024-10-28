"""
Purpose:
    Compute a political ideology score for Twitter users based on a list of domains (MBFC in this case) 
    and URLs shared by these users (along with their expanded version). A user's estimated ideology
    is calculated as the mean ideology of all domains shared (from those in the MBFC).

Inputs:
    .parquet files in data dir specified by constants (see below)

Outputs:
     .parquet file in data dir specified by constants (see below)
     - Rows represent users and columns are:
        - "user_id": Twitter ID of the user
        - "political_score": avg. political score of political URLs shared
        - "no_political_urls": number of political URLs shared

Authors:
    Francesco Pierri
"""

import glob
import os
import tldextract

import numpy as np
import pandas as pd

from collections import defaultdict

MBFC_FILE = "../../data/mbfc.csv"
POL_SCORE_FILE = "../../data/user_political_score.parquet"
URL_DIR = "/data_volume/midterm_data/entities/urls"
EXP_URL_DIR = "/data_volume/midterm_data/entities/expanded_urls"

BIAS_MAPPING = {
    'LEFT-CENTER': -0.33,
    'RIGHT-CENTER': +0.33,
    'LEFT': -0.66,
    # 'LEFT-CENTER (by Saudi standards)': -0.33,
    'LEFT CENTER': -0.33,
    'RIGHT': +0.66,
    'FAR RIGHT': +1,
    'RIGHT CONSPIRACY- PSEUDOSCIENCE': +0.66,
    'RIGHT CONSPIRACY-PSEUDOSCIENCE': +0.66,
    'RIGHT-CONSPIRACY': +0.66,
    'RIGHT PSEUDOSCIENCE': +0.66,
    'RIGHT-CONSPIRACY-PSEUDOSCIENCE': +0.66,
    'EXTREME RIGHT': +1, 
    'RIGHT CONSPIRACY': +0.66,
    'PSEUDOSCIENCE RIGHT BIASED': +0.66,
    'RIGHT-PSEUDOSCIENCE': +0.66,
    'RIGHT CONSPIRACY – PSEUDOSCIENCE': +0.66,
    'FAR RIGHT CONSPIRACY-PSEUSDOSCIENCE': +1,
    'LEFT CONSPIRACY-PSEUDOSCIENCE': -0.66,
    'LEFT-PSEUDOSCIENCE': -0.66,
    'FAR-RIGHT CONSPIRACY-PSEUDOSCIENCE': +1,
    'LEFT-CONSPIRACY/PSEUDOSCIENCE': -0.66,
    'ALT-RIGHT CONSPIRACY': +0.66,
    'RIGHT-PSUEDOSCIENCE': +0.66,
    'EXTREME RIGHT CONSPIRACY': +1,
    'LEFT – PSEUDOSCIENCE': -0.66,
    'RIGHT – CONSPIRACY AND PSEUDOSCIENCE': +0.66,
    'EXTREME LEFT': -1,
    'RIGHT – CONSPIRACY': +0.66,
    'RIGHT-CONSPIRACY/PSEUDOSCIENCE': +0.66,
    'LEFT PSEUDOSCIENCE': -0.66,
    'RIGHT CONSPIRACY/PSEUDOSCIENCE': +0.66,
    'EXTREME-RIGHT': +1,
    'FAR-RIGHT': +1,
    'FAR RIGHT-BIAS': +1,
    'FAR LEFT': -1, 
    'EXTREME-LEFT': -1,
    'FAR LEFT BIAS': -1,
    'PRO-SCIENCE (LEFT-LEANING)':-0.66,
    'LEFT-CENTER – PRO-SCIENCE': -0.33,
    'LEFT-CENTER PRO-SCIENCE': -0.33,
    'LEFT LEANING PRO-SCIENCE': -0.66,
    'LEFT SATIRE': -0.66,
    'RIGHT-SATIRE': +0.66, 
    'LEFT-SATIRE': -0.66,
    'LEFT LEANING': -0.66,
    'RIGHT SATIRE': +0.66,
    'SATIRE (Left-Leaning)': -0.66,
    'LEFT BIASED': -0.66,
    'FAR-LEFT': -1,
    'LEAST BIASED': 0,
    'LEAST – PRO SCIENCE': 0,
    'LEAST PRO-SCIENCE': 0
}

def extract_top_domain(url):
    """
    Extract the top-level domain of a given URL

    Parameters:
    ----------
    url (str): URL to extract the top-level domain from

    Returns:
    -------
    str: Top-level domain of the URL guaranteed to be lowercase
    """
    extraction_result = tldextract.extract(url)
    domain = extraction_result.domain
    suffix = extraction_result.suffix
    return f"{domain}.{suffix}".lower()


def get_MBFC_score():
    """
    Take the MBFC list and create a dictionary to map domains to their score.

    Parameters:
    ----------
    - None

    Returns:
    -------
    - MBFC_score (Dict[{str: int}]): Maps Top-level domains in the MBFC list to a political score

    """
    # read the MBFC list of domains from 2022
    df = pd.read_csv(MBFC_FILE)
    df = df[["domain", "bias_rating"]]

    # exclude non political websites
    df = df[df["bias_rating"].isin(BIAS_MAPPING)]

    # excluding additional websites that do have a bias rating
    domains_to_filter = ["youtube.com",
                         "blogspot.com",
                         "facebook.com",
                         "apple.com",
                         "medium.com",
                         "house.gov"   
                        ]

    df = df[~df["domain"].isin(domains_to_filter)]
    
    # map them to a score in the range [-1, +1]
    MBFC_score = {row["domain"]: BIAS_MAPPING[row["bias_rating"]] for _, row in df.iterrows()}
    

    return MBFC_score


def compute_ideology(domain_score):
    """
    Generate and save dataframe with estimated mean political score of users
    that shared URLs from from the MBFC domain list.

    Parameters:
    ----------
    - domain_score (Dict[{str: int}]) : Maps Top-level domains in the MBFC list to a political score

    Returns:
    -------
    - None
    - Saves a .parquet pandas dataframe with the three columns:
        - user_id (str) : twitter unique user id
        - political_score (float) : estimated political ideology score. Represents the mean
            of all shared domains from the MBFC domain list.
        - no_political_urls (int) : number of political urls shared by `user_id`
    """
    user_url_dict = defaultdict(list)
    # Looping over URLs files, example of filename: urls--2022-10-05.parquet
    for file in glob.glob(os.path.join(URL_DIR, "*2022*")):
        print(file)
        url_df = pd.read_parquet(file)

        # Exclude quotes, which do not necessarily indicate endorsement
        url_df = url_df[url_df["from_quoted_status"] == False]

        # Drop unnecessary columns
        url_df = url_df[["post_id", "user_id", "raw_url"]]

        # Reading associated file with expanded URLs, example of filename: expanded_url--2022-10-19.parquet
        date = os.path.basename(file).split("--")[1].rstrip(".parquet")
        exp_file = os.path.join(EXP_URL_DIR, "expanded_url--" + date + ".parquet")

        if os.path.exists(exp_file):
            exp_url_df = pd.read_parquet(exp_file)

            # Dropping unnecessary columns
            exp_url_df = exp_url_df[["post_id", "expanded_url"]]

            # Merging the two files
            data = url_df.merge(exp_url_df[["post_id", "expanded_url"]])
        else:
            data = url_df

        # Updating the list of URLs shared by users
        for ix, row in data.iterrows():
            url = row["expanded_url"] if "expanded_url" in row else row["raw_url"]
            user_id = row["user_id"]
            user_url_dict[user_id].append(url)


    # For all users, generate a list of ideology scores for each
    # domain shared by a user that is in our list of domains
    user_score = []
    for user in user_url_dict:
        score = []
        for url in user_url_dict[user]:

            # Extracting the Top-Level Domain of the URL
            domain = extract_top_domain(url)
            if domain in domain_score:
                score.append(domain_score[domain])

        # Calculate average score, if there is at least one tweet with domain from our list
        if score:
            user_score.append(
                {
                    "user_id": user,
                    "political_score": np.mean(score),
                    "no_political_urls": len(score),
                }
            )
    user_score = pd.DataFrame(user_score)
    user_score.to_parquet(POL_SCORE_FILE, index=False, engine="pyarrow")


if __name__ == "__main__":

    domain_score = get_MBFC_score()
    compute_ideology(domain_score)
