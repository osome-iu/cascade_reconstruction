#!/bin/bash

# This script that was utilized to generate the python virtual environment.
# 
# Note: conda (V 24.4.0) was installed already
# 
# Author: Matthew DeVerna

# Upgrade pip to the latest version
pip install --upgrade pip

# Install the required packages with specified versions
pip install igraph==0.11.4 \
            joblib==1.4.0 \
            matplotlib==3.8.4 \
            numpy==1.26.4 \
            pandas==2.2.2 \
            pyarrow==16.0.0 \
            scipy==1.13.0 \
            seaborn==0.13.2 \
            statsmodels==0.14.2

echo ""
echo "Environment creation completed."


