"""
Purpose:
    Generate bibliographic statistics from OpenAlex clean records.

Inputs:
    - ../info_diff_review/data/open_alex_clean_records.parquet: Parquet file containing bibliographic records.

Outputs:
    - statistics/bibliographic_stats.txt: Text file with bibliographic statistics.

Author:
    Matthew DeVerna
"""

import os
import pandas as pd

OUTPUT_DIR = "statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FN = "bibliographic_stats.txt"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_parquet("../info_diff_review/data/open_alex_clean_records.parquet")

# Convert the publication date to a datetime object
df["pub_date"] = pd.to_datetime(df["pub_date"])
df = df.sort_values("pub_date")

# Things we don't want!
is_pre_2006 = df.pub_year < 2006
is_preprint = df.pub_version == "submittedVersion"
is_not_article = df.pub_type != "article"
is_not_journo_conf = ~df.pub_subtype.isin(["journal", "conference"])

num_pre_2006_papers = sum(is_pre_2006)
num_preprints = sum(is_preprint)
num_non_articles = sum(is_not_article)
num_non_journo_confs = sum(is_not_journo_conf)
with open(os.path.join(OUTPUT_DIR, OUTPUT_FN), "w") as f:
    f.write("BIBLIOGRAPHIC STATISTICS\n\n")
    f.write(f"Total number of papers: {len(df):,}\n\n")
    f.write("REMOVING UNWANTED DATA\n")
    f.write("-" * 25 + "\n")
    f.write("Note: The below category counts may not be mutually exclusive.\n")
    f.write(f"Number of papers from before 2006: {num_pre_2006_papers:,}\n")
    f.write(f"Number of preprint papers        : {num_preprints:,}\n")
    f.write(f"Number of papers not articles.   : {num_non_articles:,}\n")
    f.write(f"Number of papers not journo/confs: {num_non_journo_confs:,}\n\n")

clean_df = df[
    ~is_pre_2006 & ~is_preprint & ~is_not_article & ~is_not_journo_conf
].reset_index(drop=True)

total_items = len(clean_df)
clean_df = clean_df.dropna(subset=["doi", "pub_title", "field"]).reset_index(drop=True)

err_msg = "Error: non-peer-reviewed data points found."
assert ["journal", "conference"] == clean_df.pub_subtype.unique().tolist(), err_msg
with open(os.path.join(OUTPUT_DIR, OUTPUT_FN), "a") as f:
    f.write("Only peer-reviewed work remains.\n\n")
    f.write("---\n")
    f.write(f"Number of papers with NA         : {total_items - len(clean_df):,}\n")
    f.write(f"Total number of rows removed     : {len(df) - len(clean_df):,}\n")
    f.write("-" * 25 + "\n\n")

# Show count by field and proportion of "Other" fields relative to all fields
count_by_field = clean_df.value_counts("field").to_frame("num_pubs").reset_index()

# Write the value counts to the output file
with open(os.path.join(OUTPUT_DIR, OUTPUT_FN), "a") as f:
    f.write("Count by Field\n")
    f.write("-" * 25 + "\n")
    for _, row in count_by_field.iterrows():
        f.write(f"{row['field']}: {row['num_pubs']:,}\n")
    f.write("\n")

topfields = count_by_field.head(15).copy()
num_in_top_fifteen = topfields["num_pubs"].sum()
percent_in_top_fifteen = (num_in_top_fifteen / count_by_field["num_pubs"].sum()) * 100

with open(os.path.join(OUTPUT_DIR, OUTPUT_FN), "a") as f:
    f.write(f"% of publications in top fifteen fields: {percent_in_top_fifteen:.2f}%\n")
    f.write(
        f"% of publications in bottom ten fields: {100-percent_in_top_fifteen:.2f}%\n"
    )
