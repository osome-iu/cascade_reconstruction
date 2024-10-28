"""
Convenience utility functions for the portion of this project related to midterm data.
"""

import datetime
import fnmatch
import glob
import logging
import os
import sys

import pandas as pd

TWITTER_DATE_STRING_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def get_files_in_date_range(start_date, end_date, data_directory):
    """
    Get a list of file names in a specified date range from a given directory.
    Files must match the following string pattern: "streaming_data--*.json.gz"

    Parameters:
    ------------
    - start_date (str): The start date in 'YYYY-MM-DD' format.
    - end_date (str): The end date in 'YYYY-MM-DD' format.
    - data_directory (str): The directory where files are located.

    Returns:
    ------------
    - list: Sorted list of file names within the specified date range.
    """

    # Create list of formatted dates for our date range
    date_range = pd.date_range(start=start_date, end=end_date)
    formatted_dates = [date.strftime("%Y-%m-%d") for date in date_range]

    # Construct the glob pattern
    glob_pattern = os.path.join(data_directory, "streaming_data--*.json.gz")

    # Select only files for this date range
    all_files = glob.glob(glob_pattern)
    filtered_files = []
    for file in all_files:
        for date in formatted_dates:
            if fnmatch.fnmatch(file, f"*{date}*.json.gz"):
                filtered_files.append(file)

    # Sort the list of filtered files
    filtered_files.sort()
    return filtered_files


def convert_string_to_datetime(date_string):
    """
    Convert a date string of the format 'Thu Dec 29 23:49:35 +0000 2022' to a datetime object.

    Parameters:
    ------------
    - date_string (str): The date string to be converted.

    Returns:
    ------------
    - datetime: The datetime object representing the given date string.
    """
    format_string = TWITTER_DATE_STRING_FORMAT
    return datetime.datetime.strptime(date_string, format_string).replace(tzinfo=None)


def get_dict_val(dictionary: dict, key_list: list = []):
    """
    Return `dictionary` value at the end of the key path provided in `key_list`.
    Indicate what value to return based on the key_list provided. For example, from
    left to right, each string in the key_list indicates another nested level further
    down in the dictionary. If no value is present, `None` is returned.

    Parameters:
    ----------
    - dictionary (dict) : the dictionary object to traverse
    - key_list (list) : list of strings indicating what dict_obj
        item to retrieve

    Returns:
    ----------
    - key value (if present) or None (if not present)
    Raises:

    - TypeError
    Examples:
    ---------
    # Create dictionary
    dictionary = {
        "a" : 1,
        "b" : {
            "c" : 2,
            "d" : 5
        },
        "e" : {
            "f" : 4,
            "g" : 3
        },
        "h" : 3
    }
    ### 1. Finding an existing value
    # Create key_list
    key_list = ['b', 'c']
    # Execute function
    get_dict_val(dictionary, key_list)
    # Returns
    2
    ~~~
    ### 2. When input key_path doesn't exist
    # Create key_list
    key_list = ['b', 'k']
    # Execute function
    value = get_dict_val(dictionary, key_list)
    # Returns NoneType because the provided path doesn't exist
    type(value)
    NoneType
    """
    if not isinstance(dictionary, dict):
        raise TypeError("`dictionary` must be of type `dict`")

    if not isinstance(key_list, list):
        raise TypeError("`key_list` must be of type `list`")

    retval = dictionary
    for k in key_list:
        # If retval is not a dictionary, we're going too deep
        if not isinstance(retval, dict):
            return None

        if k in retval:
            retval = retval[k]

        else:
            return None
    return retval


def collect_files_recursively(matching_str, dirname="./"):
    """
    Return all files under `dirname` that match `matching_str`.

    Parameters
    ----------
    - matching_str (str) : a wildcard-friendly matching string (via fnmatch)
    - dirname (str) : the directory to search recursively (Default = "./")

    Returns
    ---------
    - files (List[str]): list of full file paths
    """
    files = []

    for dirpath, _, filenames in os.walk(dirname):
        for filename in filenames:
            if fnmatch.fnmatch(filename, matching_str):
                files.append(os.path.join(dirpath, filename))

    return files

 
def get_logger(log_dir, log_fname, also_print=False):
    """
    Create logger.

    Parameters
    ----------
    - log_dir (str): path to directory where save the logging file
    - log_fname (str) : name of logging file
    """

    # Create log_dir if it doesn't exist already
    try:
        os.makedirs(f"{log_dir}")
    except:
        pass

    # Create logger and set level
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # Configure file handler
    formatter = logging.Formatter(
        fmt="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
        datefmt="%Y-%m-%d_%H:%M:%S",
    )
    full_log_path = os.path.join(log_dir, log_fname)
    fh = logging.FileHandler(f"{full_log_path}")
    fh.setFormatter(formatter)
    fh.setLevel(level=logging.INFO)
    # Add handlers to logger
    logger.addHandler(fh)

    # If also_print is true, the logger will also print the output to the
    # console in addition to sending it to the log file
    if also_print:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level=logging.INFO)
        logger.addHandler(ch)

    return logger
