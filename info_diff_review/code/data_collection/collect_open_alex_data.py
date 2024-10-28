"""
Purpose:
    Collect scholarly data from OpenAlex related to information diffusion and social media.

Input:
    - None.
    - Search terms and operators, as well paths, are set below with constants.

Output:
    - .jsonl files are saved where each line represents one result from OpenAlex.
    - results are altered to include the query parameter and the time the query was made,
        otherwise they represent the raw data from OpenAlex.

Authors:
- Matthew DeVerna
"""

import datetime
import json
import os
import requests
import time

QUERY = (
    '("information diffusion" OR "diffusion of information" OR "information spread" OR "spread of information") '
    'AND ("social media" OR "facebook" OR "twitter" OR "reddit")'
)
OUTPUT_DIR = "/data_volume/cascade_reconstruction/open_alex"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def search_works_openalex(query):
    """
    Searches the OpenAlex "works" database based on a given
    query within the provided search operator.
    Utilizes cursor-based pagination to retrieve all matching records.

    Reference: https://docs.openalex.org/api-entities/works/search-works

    Parameters:
    -----------
    - query (str): query to search within search operator.

    Returns:
    -----------
    - results : All works matching the query.
    """
    base_url = "https://api.openalex.org/works"
    headers = {"Accept": "application/json"}
    results = []

    print(f"Boolean search: `{query}`")
    cursor = "*"  # Initial cursor value for pagination
    try:
        while cursor:
            params = {
                "search": query,
                "cursor": cursor,
                "mailto": "mdeverna@iu.edu",
                "per-page": 200,
            }
            query_timestamp = int(datetime.datetime.now().timestamp())
            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                # Add query and query time to each result
                for item in data["results"]:
                    item.update(
                        {
                            "query": query,
                            "query_timestamp_sec": query_timestamp,
                        }
                    )
                results.extend(data["results"])
                cursor = data.get("meta", {}).get("next_cursor", None)
                if cursor:
                    print(
                        f"Cursor: {cursor} - Retrieved {len(data['results'])} records"
                    )
                else:
                    print("No more data available.")
            else:
                print(f"Failed to retrieve data: {response.status_code}")
                break
            time.sleep(1)  # Respect rate limits
    except KeyboardInterrupt as e:
        pass

    return results


if __name__ == "__main__":

    results = search_works_openalex(query=QUERY)
    print("#" * 50)
    print("Data collection completed.")
    print("#" * 50)
    fpath = os.path.join(OUTPUT_DIR, "open_alex_raw.jsonl")
    with open(fpath, "w") as f:
        for record in results:
            json_line = f"{json.dumps(record)}" + "\n"
            f.write(json_line)

    print("--- Script complete ---")
