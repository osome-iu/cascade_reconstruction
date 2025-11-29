#!/bin/bash

# Purpose:
#   Run the complete analysis pipeline for the cascade reconstruction project.
#
# Prerequisites:
#   1. Conda environment 'cascades' must be activated
#   2. Local packages must be installed (code/package/ and midterm/code/package/)
#   3. cascade_reconstruction.tar.gz decompressed to /data_volume/cascade_reconstruction/
#
# How to call:
#   conda activate cascades
#   bash run_pipeline.sh
#
# Author: Matthew DeVerna

set -e

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

echo "=========================================="
echo "CASCADE RECONSTRUCTION PIPELINE"
echo "=========================================="
echo ""
echo "🚨🚨🚨🚨 WARNING: This pipeline will:"
echo "   - Take days to weeks to complete"
echo "   - Generate millions of files (36M+ for Vosoughi data alone)"
echo "   - Require substantial disk space and inode resources"
echo "   - IF YOU DO NOT HAVE SUFFICIENT RESOURCES, THIS PIPELINE WILL CRASH YOUR SYSTEM"
echo ""
echo "Sleeping for 30 seconds to allow cancellation..."
echo "Type Ctrl+C to cancel."
sleep 30

echo ""
echo "Beginning analysis pipeline..."
echo ""

# Run Vosoughi Analysis
bash code/data_analysis/run_vosoughi_analysis.sh

# Run Bluesky Analysis
bash bluesky/code/data_analysis/run_bsky_analysis.sh

# Run Midterm Analysis
bash midterm/code/data_analysis/run_midterm_analysis.sh

# Generate Figures and Statistics
bash generate_figures_and_statistics.sh

echo ""
echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
