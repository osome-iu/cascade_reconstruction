"""
Purpose:
    Clean the downloaded OpenAlex data.

Input:
    - None.
    - Paths are set below with constants.

Output:
    - a single .parquet file containing the below columns:
        - doi (str): the digital object identifier URL
        - pub_title (str): the title of the publication
        - primary_location (str): the location (e.g., journal) that published this work
        - primary_topic (str): the primary topic as returned by OpenAlex
            - Reference: https://docs.openalex.org/api-entities/topics
        - subfield (str): the subfield listed within OpenAlex's primary topic field.
            - Reference: https://docs.openalex.org/api-entities/topics
        - field (str): the field listed within OpenAlex's primary topic field.
            - Reference: https://docs.openalex.org/api-entities/topics
        - domain (str): the domain listed within OpenAlex's primary topic field.
            - Reference: https://docs.openalex.org/api-entities/topics
        - authors (str): the author names. If more than one, they are concatenated
            with pipes (i.e., -> "|")
       - pub_year (int): the year of the publication
       - pub_date (str): date of publication in YYYY-mm-dd format
       - citation_count (int): number of citation that the publication has received
       - pub_type (str): type of publication (mostly 'article')
            - Reference: https://docs.openalex.org/api-entities/works/work-object#type
       - pub_subtype (str): type of pulication with a bit more detail (differentiates
            between 'conference' and 'journal')
            - Reference: https://docs.openalex.org/api-entities/works/work-object#type
       - pub_version (str) : if == 'submittedVersion' it represents a preprint.
            - Reference: https://docs.openalex.org/api-entities/works/work-object#type
       - language (str): publication language
       - filter_query (str): the query passed to the OpenAlex API
       - query_timestamp_sec (int): second-resolution timestamp of when the query was made

Authors:
- Matthew DeVerna
"""

import os
import json
import pandas as pd

from midterm.utils import get_dict_val

# Ensure we are in the scripts directory for relative paths
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

DATA_DIR = "/data_volume/cascade_reconstruction/open_alex/"
OUTPUT_DIR = "../../data"

DATA_FIELDS = {
    "doi": ["doi"],
    "pub_title": ["title"],
    "primary_location": ["primary_location", "source", "display_name"],
    "primary_topic": ["primary_topic", "display_name"],
    "subfield": ["primary_topic", "subfield", "display_name"],
    "field": ["primary_topic", "field", "display_name"],
    "domain": ["primary_topic", "domain", "display_name"],
    "authors": ["authorships"],
    "pub_year": ["publication_year"],
    "pub_date": ["publication_date"],
    "citation_count": ["cited_by_count"],
    "pub_type": ["type"],
    "pub_subtype": ["primary_location", "source", "type"],
    "pub_version": ["primary_location", "version"],
    "language": ["language"],
    "filter_query": ["filter_query"],
    "query_timestamp_sec": ["query_timestamp_sec"],
}

print("Loading records...")
records = []
for file in os.listdir(DATA_DIR):
    fpath = os.path.join(DATA_DIR, file)
    with open(fpath, "r") as f:
        for line in f:
            records.append(json.loads(line))
print(f"Number of records found: {len(records)}")

print("Cleaning records...")
clean_records = []
for record in records:
    temp_record = {}
    for name, data_field_path in DATA_FIELDS.items():

        data = get_dict_val(record, data_field_path)
        if name == "authors":
            data = [get_dict_val(author, ["author", "display_name"]) for author in data]
            data = "|".join(data)
        temp_record.update({name: data})
    clean_records.append(temp_record)
records_df = pd.DataFrame.from_records(clean_records)

print(f"Saving data here: {OUTPUT_DIR}")
out_path = os.path.join(OUTPUT_DIR, "open_alex_clean_records.parquet")
records_df.to_parquet(out_path, index=False)
