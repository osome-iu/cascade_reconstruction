#!/bin/bash

# Purpose:
#   Run the Vosoughi data analysis pipeline. See individual scripts for details.
#
# Inputs:
#   None
#
# Output:
#   See individual scripts for information about their respective outputs.
#
# How to call:
#   ```
#   bash run_vosoughi_analysis.sh
#   ```
#
# Author: Matthew DeVerna

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Run the Python scripts
python reconstruct_all_cascades.py --type vosoughi; echo "reconstruct_all_cascades.py completed."
python reconstruct_cascades_time_inferred_diffusion.py; echo "reconstruct_cascades_time_inferred_diffusion.py completed."
python calculate_cascade_metrics.py; echo "calculate_cascade_metrics.py completed."
python calculate_cascade_metrics_tid.py; echo "calculate_cascade_metrics_tid.py completed."
python calculate_cascade_similarity_metrics.py; echo "calculate_cascade_similarity_metrics.py completed."

echo "All Vosoughi scripts completed."
echo "\n\n\n"