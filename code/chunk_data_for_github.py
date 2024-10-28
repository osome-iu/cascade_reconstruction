"""
Purpose: Chunk replication data into smaller files for Github upload.

Input (by command line argument): 
- A file path (--file-name)
- Maximum file size in MBs (--max-mb)

Output: 
- The specified file, chunked into smaller files.
- The file will be saved in a directory located where the file is found. 
- The directory will be the same as the file, minus the file extension.
- The chunked files will have the following pattern: f"{basename}_{str(fnum).zfill(5)}.{fname_ext}".
    - basename is the file name without the extension
    - fnum is the chunk number
    - fname_ext is the file extension

Author: Matthew DeVerna
"""
import argparse
import glob
import os
import time
import sys

import numpy as np
import pandas as pd

MB_MULTIPLIER = 10**-6  # Multiply num bytes by this to get MB


def parse_cl_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split the specified file into smaller files"
        " based on the provided max file size."
    )

    parser.add_argument(
        "--file-name",
        type=str,
        required=True,
        help="Full path to the file that will be chunked.",
    )
    parser.add_argument(
        "--max-mb",
        type=int,
        default=100,
        help="Maximum file size in megabytes. Default is 100.",
    )

    return parser.parse_args()


def read_data(file_name, basename, fname_ext):
    print(f"Reading ***{basename}*** data...")
    if fname_ext == "csv":
        data = pd.read_csv(file_name)
    elif fname_ext == "txt":
        with open(file_name, "r") as f:
            data = f.readlines()
    else:
        sys.exit(
            "ERROR: Can only handle csv and txt files at the moment."
            f" A {fname_ext} file was provided."
        )
    return data


def split_list_into_chunks(input_list, max_allowed_size_mb):
    # Convert max_allowed_size from megabytes to bytes
    buffer = 5_000_000  # Ensures we don't get stuck in a loop
    max_allowed_size = (max_allowed_size_mb / MB_MULTIPLIER) - buffer
    # print(f"max_allowed_size: {max_allowed_size}")

    # Initialize variables
    chunks = []  # List to store the chunks
    current_chunk = []  # Current chunk being constructed
    current_size = 0  # Size of the current chunk

    for item in input_list:
        item_size = len(
            item.encode("utf-8")
        )  # Calculate the size of the string in bytes

        # If adding the current item to the chunk doesn't exceed the maximum size, add it
        if current_size + item_size <= max_allowed_size:
            current_chunk.append(item)
            current_size += item_size
        else:
            # If adding the current item would exceed the maximum size, start a new chunk
            chunks.append(current_chunk)
            current_chunk = [item]
            current_size = item_size

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def save_data(file_name, fname_ext, data):
    print("Saving data...")
    if fname_ext == "csv":
        data.to_csv(file_name, index=False)
    elif fname_ext == "txt":
        with open(file_name, "w") as f:
            f.writelines("\n".join(data))
    else:
        sys.exit(
            "ERROR: Can only handle csv and txt files at the moment."
            f" A {fname_ext} file was provided."
        )
    print("Success.")


if __name__ == "__main__":
    print("Parsing command line arguments...")
    args = parse_cl_args()
    file_name = args.file_name
    max_mb = args.max_mb

    print("Setting up output directories...")
    basename = os.path.basename(file_name)
    basename_no_ext, fname_ext = basename.split(".")
    file_dir = os.path.dirname(os.path.abspath(file_name))
    chunked_dir = os.path.join(file_dir, f"chunked_{basename_no_ext}")
    os.makedirs(chunked_dir, exist_ok=True)

    data = read_data(file_name, basename, fname_ext)

    # Resave the data as it is with the chunked name
    chunks = 1
    out_file_path = os.path.join(
        chunked_dir, f"{basename}_{str(chunks).zfill(5)}.{fname_ext}"
    )
    save_data(file_name=out_file_path, fname_ext=fname_ext, data=data)

    # Now we iteratively check the file size to see if it needs to be halved for GitHub
    print(f"Maximum file size allowed is {max_mb}mb")
    file_size_bytes = os.stat(out_file_path).st_size
    max_file_size_mb = file_size_bytes * MB_MULTIPLIER

    while max_file_size_mb > max_mb:
        print(f"Current max file size is {max_file_size_mb}mb.")
        print("Splitting the data...")
        chunks += 1

        if fname_ext == "csv":
            print(f"... into {chunks} chunks...")
            split_data = np.array_split(data, chunks)
        elif fname_ext == "txt":
            split_data = split_list_into_chunks(data, max_mb)

        # Iterate through the chunks and save them accordingly
        for fnum, data_chunk in enumerate(split_data, start=1):
            out_file_path = os.path.join(
                chunked_dir, f"{basename}_{str(fnum).zfill(5)}.{fname_ext}"
            )

            save_data(file_name=out_file_path, fname_ext=fname_ext, data=data_chunk)

        file_sizes = []
        for file in glob.glob(os.path.join(chunked_dir, f"{basename}*.{fname_ext}")):
            print(f"Checking file: {os.path.basename(file)}")
            file_size = os.stat(file).st_size * MB_MULTIPLIER
            print(f"\tfile size: {file_size}mb")
            file_sizes.append(file_size)

        max_file_size_mb = max(file_sizes)

    print(f"Largest file size is now {max_file_size_mb}mb.")
    print("--- Script complete. ---")
