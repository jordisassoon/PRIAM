#!/bin/bash

# Train + predict with RF using the real pollen/climate dataset
python main.py \
  --train_climate './data/train/AMPD_cl_CHELSA.csv' \
  --train_pollen './data/train/AMPD_po.csv' \
  --test_pollen './data/test/scrubbed_PATAM.csv' \
  --model MAT \
  --target PANN \
  --output_csv ./out/test.csv

# Visualize predictions
python visuals/plot.py \
  --predictions_csv ./out/test.csv \
  --output_file ./out/test.png \
  --title 'MAT Reconstruction of TANN'
