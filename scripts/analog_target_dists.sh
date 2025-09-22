#!/bin/bash

# Hardcoded shell script to run the pollen plotting Python script

POLLEN_FILE="data/train/AMPD_po.csv"
CLIMATE_FILE="data/train/AMPD_cl_CHELSA.csv"
OUTPUT_FILE="out/pollen_dist.png"
TAXA="Amaranthaceae"
TARGET="PANN"

python visuals/distribution_per_target.py "$POLLEN_FILE" "$CLIMATE_FILE" --taxa "$TAXA" --bins 25 --target-col "$TARGET" --output-file out/pollen_dist.png