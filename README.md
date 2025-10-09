# 🌿 PRISM: A tool for Palaeoclimatic Reconstruction & Interactive Statistical Modelling 

PRISM is an interactive tool for proxy-based climate reconstruction and analysis. It allows users to import palaeoecological proxy data—such as fossil pollen, sediments, or other biological indicators—and apply multiple statistical and machine learning methods to reconstruct past climate conditions. The app provides instant model training, validation, and visualization, offering insights into model performance, feature importance, and uncertainty. Users can explore reconstructed climate variables through dynamic plots, comparative analyses, and summary statistics, making it a comprehensive platform for both research and teaching in palaeoclimatology.

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

## 👩‍🔬 Authors

Dael Sassoon
 — Palaeoecologist and Marie Curie Research Fellow (GEO3BCN-CSIC, Barcelona)
Specialising in tropical palaeoecology and quantitative reconstruction of past climates. Developer of PRISM, integrating palaeoecological data with statistical and machine learning models for climate reconstruction.

Jordan Sassoon
 — Computer Scientist 
Focused on applied AI, data systems, and scientific visualization. Lead developer of the PRISM interface, integrating backend modelling pipelines with an interactive analytical framework.

## 🧠 Citation

If you use PRISM in your research, please cite as:

Sassoon, D., & Sassoon, J. (2025). PRISM: A tool for Palaeoclimatic Reconstruction & Interactive Statistical Modelling. GitHub repository: https://github.com/jordisassoon/PyLaeo
