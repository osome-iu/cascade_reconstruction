"""
Purpose: Sample cascades from the midterm data.

Notes:
    - We only consider cascades that were originated between 2022-11-02 and 2022-11-08,
        howevever, to obtain an accurate estimate of the cascade length, we consider
        data up to one week later, i.e., 2022-11-15.

Input: None
    - Reads data files based on hardcoded path (see constants below)

Output: Sample of SAMPLE_SIZE cascades (see constants below)
    - Filename: `sampled_cascades.txt`
        - Each line contains a single cascade ID
        - All sampled cascades have a minimum size of 2 (at least one retweet)

Author: Matthew DeVerna
"""

import datetime
import gzip
import json
import os

import pandas as pd

from collections import defaultdict

# Local package
from midterm.utils import get_files_in_date_range, convert_string_to_datetime


# Ensure current directory is the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "/data_volume/midterm_data/streaming_data"
OUTPUT_DIR = "../../data/intermediate_files/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date range for files read
FILES_START_DATE = "2022-11-02"
FILES_END_DATE = "2022-11-15"

# Date range between which to select cascades
CAS_START_DATE = "2022-11-02"
CAS_END_DATE = "2022-11-08"

MIN_CAS_SIZE = 2
SAMPLE_SIZE = 10_000


def date_between(target_date, start_date_str, end_date_str):
    """
    Check if a datetime object is between two dates.

    Parameters:
    -----------
    - target_date (datetime.datetime): The datetime object to check.
    - start_date_str (str): The start date in 'YYYY-MM-DD' format.
    - end_date_str (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    -----------
    - bool: True if target_date is between start and end dates, inclusive. False otherwise.
    """
    # Convert string dates to datetime objects
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

    # Ensure tweets on the last date are included.
    # Currently the format is datetime.datetime(YYYY, MM, DD, 0, 0) (bc the
    # hours and seconds are not included in the str), so the end date is excluded
    end_date = end_date + datetime.timedelta(days=1)

    # Check if the target date is between the start and end dates (inclusive)
    return start_date <= target_date <= end_date


def get_cascade_id_counter(filtered_files):
    """
    Get a dictionary of cascade IDs and their corresponding retweet counts.

    Parameters:
    -----------
    - filtered_files (list): A list of filtered file names.

    Returns:
    -----------
    - dict: A dictionary of cascade IDs and their corresponding retweet counts.
    """
    cascade_id_counter = defaultdict(int)
    for file in filtered_files:
        print(f"Loading {file}...")
        try:
            with gzip.open(file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if "retweeted_status" in data:
                        rt_status = data["retweeted_status"]

                        # Ensure creation date is within our date range
                        rt_creation_dt = convert_string_to_datetime(
                            rt_status["created_at"]
                        )
                        if date_between(rt_creation_dt, CAS_START_DATE, CAS_END_DATE):
                            curr_cas_id = rt_status["id_str"]
                            curr_cas_rt_count = rt_status["retweet_count"]
                            prev_cas_rt_count = cascade_id_counter[curr_cas_id]
                            max_cas_rt_count = max(prev_cas_rt_count, curr_cas_rt_count)
                            cascade_id_counter[curr_cas_id] = max_cas_rt_count
                    else:
                        # We know these creation dates
                        curr_cas_id = data["id_str"]
                        curr_cas_rt_count = data["retweet_count"]
                        prev_cas_rt_count = cascade_id_counter[curr_cas_id]
                        max_cas_rt_count = max(prev_cas_rt_count, curr_cas_rt_count)
                        cascade_id_counter[curr_cas_id] = max_cas_rt_count

        except Exception as e:
            print("Error processing file: ", file)
            print(e)
    return dict(cascade_id_counter)


if __name__ == "__main__":
    print("Selecting correct files...")
    print(f"\t - Start date: {FILES_START_DATE}")
    print(f"\t - End date: {FILES_END_DATE}")
    filtered_files = get_files_in_date_range(FILES_START_DATE, FILES_END_DATE, DATA_DIR)

    print("Building cascade id counter...")
    print("\t - Considering cascades between...")
    print(f"\t\t - Start date: {CAS_START_DATE}")
    print(f"\t\t - End date: {CAS_END_DATE}")
    cascade_id_counter = get_cascade_id_counter(filtered_files)

    print("Converting to dataframe...")
    cascade_id_records = [
        {"cascade_id": key, "retweet_count": val}
        for key, val in cascade_id_counter.items()
    ]
    df = pd.DataFrame.from_records(cascade_id_records)

    print(f"Selecting cascades that have a size of at least {MIN_CAS_SIZE}...")
    num_retweets = MIN_CAS_SIZE - 1
    df = df[df.retweet_count >= num_retweets].reset_index(drop=True)

    print(f"Sampling {SAMPLE_SIZE:,} cascades...")
    sampled_cascades = df.cascade_id.sample(SAMPLE_SIZE).values

    output_file = os.path.join(OUTPUT_DIR, "sampled_cascades.txt")
    print(f"Writing sampled cascades to {output_file}...")
    with open(output_file, "w") as f:
        for cascade_id in sampled_cascades:
            f.write(cascade_id + "\n")

    print("Script complete.")
