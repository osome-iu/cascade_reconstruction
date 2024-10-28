#!/bin/bash

# Purpose:
#   Run the midterm analysis pipeline. See individual scripts for details.
#
# Inputs:
#   None
#
# Output:
#   See the individual scripts for details on their output.
#
# How to call:
#   ```
#   bash run_midterm_analysis.sh
#   ```
#
# Author: Matthew DeVerna

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Run the Python scripts
python 001_reconstruct_all_cascades.py; echo "001_reconstruct_all_cascades.py completed."
python 002_generate_networks.py; echo "002_generate_networks.py completed."
python 003_calculate_node_centralities.py; echo "003_calculate_node_centralities.py completed."
python 004_generate_naive_network.py; echo "004_generate_naive_network.py completed."
python 005_calculate_naive_net_node_centralities.py; echo "005_calculate_naive_net_node_centralities.py completed."
python 006_calculate_centrality_jaccard.py; echo "006_calculate_centrality_jaccard.py completed."
python 007_calculate_strength_change.py; echo "007_calculate_strength_change.py completed."
python 008_calculate_naive_reconstructed_strength_correlations.py; echo "008_calculate_naive_reconstructed_strength_correlations.py completed."

echo "Midterm analysis pipeline completed."
echo "\n\n\n"