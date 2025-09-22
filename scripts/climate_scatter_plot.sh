#!/bin/bash

# Hardcoded shell script to run the pollen plotting Python script

CLIMATE_FILE="data/train/AMPD_cl_CHELSA.csv"
X_TARGET="PANN"
Y_TARGET="MTCO"
OUTPUT_FILE="out/{$X_TARGET}_vs_{$Y_TARGET}.png"

python visuals/climate_scatter.py "$CLIMATE_FILE" --x-var "$X_TARGET" --y-var "$Y_TARGET" --output-file "$OUTPUT_FILE"
