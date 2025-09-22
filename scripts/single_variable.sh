#!/bin/bash

# Hardcoded shell script to run the pollen plotting Python script

CLIMATE_FILE="data/train/AMPD_cl_CHELSA.csv"
VAR="TANN"
OUTPUT_FILE="out/$VAR.png"

python visuals/single_variable.py "$CLIMATE_FILE" --var "$VAR" --output-file "$OUTPUT_FILE"
