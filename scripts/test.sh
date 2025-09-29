#!/bin/bash

# Train + predict with MAT using the real pollen/climate dataset
python main.py \
  --train_climate "./data/train/AMPD_cl_CHELSA.csv" \
  --train_pollen "./data/train/AMPD_po.csv" \
  --test_pollen "./data/test/scrubbed_PATAM.csv" \
  --taxa_mask "./data/train/taxa_mask.csv" \
  --model_name ALL \
  --target PANN \
  --k 3 \
  --cv_folds 5 \
  --output_csv "./out/all_predictions.csv"

# Visualize predictions
python visuals/plot.py \
  --predictions_csv "./out/all_predictions.csv" \
  --output_file "./out/all_predictions.png" \
  --title "ALL Reconstruction of PAAN" \
  --depth_csv './data/test/scrubbed_PATAM.csv'
