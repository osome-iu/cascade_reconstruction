#!/bin/bash

# Create different yaml files in all the ways the conda
#   allows, since it can be finicky depending on the system.
# Our hope is that giving you all the versions make it easier to troubleshoot
#   should you run into any issues.
# 
# Note that the environment was activated prior to running this script!
# 
# Author: Matthew DeVerna

# Change to the directory where the script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

conda env export --from-history > env_from_history_cascades.yml
conda env export > env_cascades.yml
conda list --explicit > env_explicit_cascades.txt

echo ""
echo "Environment files created."