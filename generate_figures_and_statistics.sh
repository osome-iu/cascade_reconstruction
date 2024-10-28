#!/bin/bash

# Purpose:
#   Run the figure and statistics generation pipelines.
#
# Inputs:
#   None
#
# Output:
#   Each script generates respective outputs in the statistics/ and figures/ directories.
#
# How to call:
#   ```
#   bash generate_figures_and_statistics.sh
#   ```
#
# Author: Matthew DeVerna

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Run the statistics generation pipeline
echo "BEGINNING STATISTICS GENERATION PIPELINE"
echo "#########################################"
bash generate_statistics/run_stats_gen_pipeline.sh; echo "run_stats_gen_pipeline.sh completed."
echo " "
echo " "
echo " "

# Run the figure generation pipeline
echo "BEGINNING FIGURE GENERATION PIPELINE"
echo "####################################"
bash generate_figures/run_fig_gen_pipeline.sh; echo "run_fig_gen_pipeline.sh completed."

echo "All figures and statistical outputs created."
echo " "
echo " "
echo " "