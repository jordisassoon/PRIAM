#!/bin/bash

# Train + predict with MAT using the real pollen/climate dataset
python main.py \
  --train_climate "./data/odp/ss3267cl.csv" \
  --train_pollen "./data/odp/ss3267po.csv" \
  --test_pollen "./data/odp/ODP976_all.csv" \
  --taxa_mask "./data/train/taxa_mask.csv" \
  --model_name MAT \
  --target MAP \
  --k 3 \
  --cv_folds 5 \
  --output_csv "./out/mat_predictions.csv"

# Visualize predictions
python visuals/plot.py \
  --predictions_csv "./out/mat_predictions.csv" \
  --output_file "./out/mat_predictions.png" \
  --title "MAT Reconstruction of MAP" \
  --depth_csv './data/odp/odp976 age.csv'
