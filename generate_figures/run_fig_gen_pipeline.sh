#!/bin/bash

# Purpose:
#   Run the figure generation pipeline. See individual scripts for details.
#
# Inputs:
#   None
#
# Output:
#   Each script generates a pdf, svg, and png file in the figures/ directory.
#
# How to call:
#   ```
#   bash run_fig_gen_pipeline.sh
#   ```
#
# Author: Matthew DeVerna

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Run the Python scripts
python generate_correlations_heatmap.py; echo "generate_correlations_heatmap.py completed."
python generate_top_k_influence_comparison_with_communities.py; echo "generate_top_k_influence_comparison_with_communities.py completed."
python generate_cascade_similarity_nine_panel.py; echo "generate_cascade_similarity_nine_panel.py completed."
python generate_node_feature_ccdfs.py; echo "generate_node_feature_ccdfs.py completed."
python generate_bibliographic_figure.py; echo "generate_bibliographic_figure.py completed."

echo "All figures created."
echo ""
echo ""
echo ""
