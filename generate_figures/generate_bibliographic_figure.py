"""
Purpose:
    Generate the bibliographic figure. Plots the total number of publications for the
    top 15 fields, along with an "Other" category that combines all not in the top 15.
    Also includes a time series stem plot of the number of publications per year.

Inputs:
    - None. Data read via constants/paths defined in the script.

Outputs:
    - The figure saved in PDF, PNG, and SVG formats.

Author:
    Matthew DeVerna
"""

import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Change the current working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "figures"

mpl.rcParams["font.size"] = 14
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

from matplotlib.ticker import FuncFormatter


def thousand_comma_formatter(x, pos):
    """
    Takes a tick label value and position, and returns the value
    formatted with commas for thousands.

    Parameters:
    - x: The tick label value.
    - pos: The position of the tick (not used in this formatter).

    Returns:
    - A string with the tick label value formatted with commas for thousands.
    """
    return f"{x:,.0f}"


formatter = FuncFormatter(thousand_comma_formatter)

field_abbreviations = {
    "Social Sciences": "Soc. Sci.",
    "Physics and Astronomy": "Phys. & Astron.",
    "Computer Science": "Comp. Sci.",
    "Business, Management and Accounting": "Business, Mgmt. & Acct.",
    "Medicine": "Medicine",
    "Psychology": "Psychology",
    "Economics, Econometrics and Finance": "Economics & Finance",
    "Decision Sciences": "Decisions Sciences",
    "Arts and Humanities": "Arts & Humanities",
    "Health Professions": "Health Professions",
    "Engineering": "Engineering",
    "Environmental Science": "Env. Sci.",
    "Agricultural and Biological Sciences": "Agricultural & Biol. Sci.",
    "Other": "Other",
    "Mathematics": "Mathematics",
    "Biochemistry, Genetics and Molecular Biology": "Biochem., Genet. & Mol. Biol.",
}

df = pd.read_parquet("../info_diff_review/data/open_alex_clean_records.parquet")

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

clean_df = df[
    ~is_pre_2006 & ~is_preprint & ~is_not_article & ~is_not_journo_conf
].reset_index(drop=True)

total_items = len(clean_df)
clean_df = clean_df.dropna(subset=["doi", "pub_title", "field"]).reset_index(drop=True)

err_msg = "Error: non-peer-reviewed data points found."
assert ["journal", "conference"] == clean_df.pub_subtype.unique().tolist(), err_msg

# Count the number of publications per year
by_year_count = (
    clean_df.value_counts("pub_year").sort_index().to_frame("num_pubs").reset_index()
)
by_year_count = by_year_count[by_year_count["pub_year"] < 2024]

# Create a dataframe that includes only the top 15 fields and an "Other" category
count_by_field = clean_df.value_counts("field").to_frame("num_pubs").reset_index()
topfields = count_by_field.head(15).copy()
num_in_top_ten = topfields["num_pubs"].sum()
not_topfields = count_by_field[~count_by_field["field"].isin(topfields.field.values)]
topfields = pd.concat(
    [
        topfields,
        pd.DataFrame({"field": ["Other"], "num_pubs": [not_topfields.num_pubs.sum()]}),
    ]
).reset_index(drop=True)

# Rename the fields to their abbreviations
topfields["field"] = topfields["field"].map(field_abbreviations)

print("\n")
print("Generating the figure...")

fig, ax1 = plt.subplots(figsize=(10, 7))


topfields = topfields.sort_values("num_pubs")

# Create an instance of FuncFormatter using the custom function
formatter = FuncFormatter(thousand_comma_formatter)


### Generate timeseries stem plot.
ax1.stem(by_year_count["num_pubs"], linefmt="k", markerfmt="ok", basefmt="none")

for idx, row in by_year_count.iterrows():
    ax1.text(
        x=idx,
        y=row.num_pubs + 50 if len(str(row.num_pubs)) < 4 else row.num_pubs + 70,
        s=f"{row.num_pubs:,}",
        ha="center",
        rotation=0 if len(str(row.num_pubs)) < 4 else 90,
    )

xtick_vals = (by_year_count["pub_year"] - by_year_count["pub_year"].min()).to_list()
ax1.set_xticks(xtick_vals, by_year_count["pub_year"], rotation=80)
ax1.set_ylim(0, 2250)


ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)

ax1.set_xlabel("year of publication", labelpad=10)
ax1.set_ylabel("number of publications", rotation=270, va="bottom", labelpad=15)

ax1.yaxis.set_major_formatter(formatter)
ax1.yaxis.set_ticks_position("right")
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position("right")


### Generate bar plot
ax2 = inset_axes(ax1, width="85%", height="90%", loc="upper left", borderpad=0)
ax2.barh(topfields["field"], topfields["num_pubs"], color="#990000", zorder=3)
ax2.xaxis.set_major_formatter(formatter)
ax2.tick_params(axis="y", length=0)

for idx, row in topfields.iterrows():
    ax2.text(
        x=row.num_pubs + 10, y=row.field, s=f"{row.num_pubs:,}", ha="left", va="center"
    )

ax2.set_xlim(0, 7000)
ax2.set_facecolor("none")

# ax2.spines['left'].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.spines["right"].set_visible(False)
# ax2.spines['top'].set_visible(False)

ax2.set_xlabel("number of publications", labelpad=10)
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position("top")


plt.subplots_adjust(hspace=0.25)


# Define the file name and extensions
file_name = "lit_review_ts"
extensions = ["pdf", "png", "svg"]

# Save the figure in different formats using a loop
for ext in extensions:
    output_path = os.path.join(OUTPUT_DIR, f"{file_name}.{ext}")
    fig.savefig(output_path, dpi=800, bbox_inches="tight")
    print(f"- Created: {output_path}")
