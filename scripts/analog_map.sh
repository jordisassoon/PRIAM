#!/bin/bash

# Hardcoded shell script to run the pollen plotting Python script

POLLEN_FILE="data/train/AMPD_po.csv"
COORDS_FILE="data/train/AMPD_co.csv"
OUTPUT_HTML="out/map_amazon.html"

python visuals/location_map.py "$POLLEN_FILE" "$COORDS_FILE" --output-html "$OUTPUT_HTML"