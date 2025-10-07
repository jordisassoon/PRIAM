# 🌿 PyLaeo: A Python tool for Palaeoecological Statistical Modeling

This project implements multiple methods for quantitative palaeoclimate reconstruction from fossil pollen data, including:

* **Modern Analogue Technique (MAT)**
* **Boosted Regression Trees (BRT)**
* **Weighted Averaging Partial Least Squares (WA-PLS)**
* **Random Forest (RF)**

It also provides a **web-based interface** using [Streamlit](https://streamlit.io).

## 📦 Installation

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/PyLae.git
cd pollen-recon
```

### 2. Create and activate a conda environment

```bash
conda env create -f environment.yml
conda activate pylaeo
```

## ⚙️ Command-Line Usage

The main pipeline is in `main.py`.
It trains models on **training climate + pollen data** and predicts for **fossil pollen samples**.

### Example

```bash
python main.py \
  --train_climate ./data/train/AMPD_cl_worldclim2.csv \
  --train_pollen ./data/train/AMPD_po.csv \
  --test_pollen ./data/test/scrubbed_SAR.csv \
  --model RF \
  --target TANN \
  --output_csv ./out/predictions.csv
```

Arguments:

* `--train_climate`: CSV with climate variables (targets).
* `--train_pollen`: CSV with modern pollen counts.
* `--test_pollen`: CSV with fossil pollen data.
* `--model`: Model choice (`MAT`, `BRT`, `WA-PLS`, `RF`).
* `--target`: Target variable to reconstruct (e.g., `TANN`).
* `--output_csv`: Where to save predictions.

## 📊 Visualization

Once predictions are saved, you can plot them with:

```bash
python visuals/plot.py \
  --predictions_csv ./out/predictions.csv \
  --output_file ./out/predictions.png \
  --title "Reconstructed TANN"
```

## 🌐 Streamlit Web App

For an interactive interface:

```bash
streamlit run app.py
```

This will launch a web UI at [http://localhost:8501](http://localhost:8501).

Features:

* Upload training and fossil pollen CSVs.
* Choose model + target variable.
* Run predictions interactively.
* View results as a table and time-series plot.
* Download predictions as CSV.

## 📂 Project Structure

```
├── app.py                # Streamlit web app
├── main.py               # CLI pipeline
├── models/               # Model classes (MAT, BRT, WA-PLS, RF)
├── utils/dataloader.py   # Data loading + preprocessing
├── visuals/plot.py       # Visualization script
├── data/                 # Example datasets
├── out/                  # Output predictions + plots
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## 🧑‍💻 Development Notes

* Datasets are aligned automatically (non-overlapping taxa filled with zeros).
* Predictions are saved with column name `Predicted_<target>`.
* Tested on Linux (Python 3.10).

## Coming Features

- [X] Model parameter section (BRT iteration)  
- [X] Smoothing  
- [ ] Error / trend — check existing code  
- [X] R² + RMSE metrics  
- [X] Cross-validation results / bootstrapping  
- [X] Map for analogues  
- [ ] Neighbours for analogues
- [ ] Trees visualisation
- [ ] Fix WAPLS
- [ ] FIx scatter plot (fossil layer)
- [X] Selection of taxa to include / exclude  
- [ ] Visualise imported files  
- [ ] Feature importance  
- [ ] Harmonisation checker  
- [ ] Author info
- [ ] Dummy dataset
- [ ] Saving system
- [ ] Invert X axis (age)
- [ ] Language choice (?) 
- [X] Pollen diagram (presence over time: % and counts)
- [X] Plot records on 2D Variable graph (for example TANN/PANN)
