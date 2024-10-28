#!/bin/bash

# Purpose:
#   Run the statistics generation pipeline. See individual scripts for details.
#
# Inputs:
#   None
#
# Output:
#   Each script generates a .txt file in the statistics/ directory.
#
# How to call:
#   ```
#   bash run_stats_gen_pipeline.sh
#   ```
#
# Author: Matthew DeVerna

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Run the Python scripts

python3 generate_bibliographic_stats.py ; echo "generate_bibliographic_stats.py completed."
python3 generate_ccdf_comparisons.py; echo "generate_ccdf_comparisons.py completed."
python3 generate_heatmap_corr_stats.py; echo "generate_heatmap_corr_stats.py completed."
python3 generate_influence_change_stats.py; echo "generate_influence_change_stats.py completed."

echo "All statistical outputs created."
echo ""
echo ""
echo ""