import streamlit as st
import pandas as pd
import numpy as np
from models.mat import MAT
from models.brt import BRT
from models.wa_pls import WA_PLS
from models.rf import RF
from utils.dataloader import PollenDataLoader

# App title
st.title("🌿 Pollen-based Climate Reconstruction")

# Sidebar inputs
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["MAT", "BRT", "WA-PLS", "RF", "All"],  # Add "All" option
    index=0  # Default to MAT
)
target = st.sidebar.selectbox(
    "Target climate variable",
    ["TANN", "Temp_season", "MTWA", "MTCO", "PANN"]
)
k = st.sidebar.slider("Number of neighbors (MAT only)", 1, 20, 5)
pls_components = st.sidebar.slider("PLS components (WA-PLS only)", 1, 10, 3)
random_seed = st.sidebar.number_input("Random seed", value=42)

# File uploads
st.sidebar.header("Upload Data")
train_climate_file = st.sidebar.file_uploader("Training Climate CSV", type=["csv"])
train_pollen_file = st.sidebar.file_uploader("Training Pollen CSV", type=["csv"])
test_pollen_file = st.sidebar.file_uploader("Test Fossil Pollen CSV", type=["csv"])

if train_climate_file and train_pollen_file and test_pollen_file:
    # Load data
    loader = PollenDataLoader(
        climate_file=train_climate_file,
        pollen_file=train_pollen_file,
        test_file=test_pollen_file
    )

    X_train, y_train = loader.load_training_data(target=target)
    X_test, ages = loader.load_test_data()
    X_train_aligned, X_test_aligned = loader.align_taxa(X_train, X_test)

    # Prepare models
    available_models = {
        "MAT": MAT(k=k),
        "BRT": BRT(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_seed),
        "WA-PLS": WA_PLS(n_components=pls_components),
        "RF": RF(n_estimators=100, max_depth=6, random_state=random_seed)
    }

    # Determine which models to run
    if model_choice == "All":
        models_to_run = available_models
    else:
        models_to_run = {model_choice: available_models[model_choice]}

    # Train and predict
    predictions_dict = {}
    for name, model in models_to_run.items():
        model.fit(X_train_aligned, y_train)
        predictions_dict[name] = model.predict(X_test_aligned)

    # Combine predictions into dataframe
    df_preds = pd.DataFrame({"Age": ages.values})
    for name, preds in predictions_dict.items():
        df_preds[f"{name}_{target}"] = preds

    # Display
    st.subheader("Predictions")
    st.write(df_preds)

    st.subheader("Visualization")
    st.line_chart(df_preds.set_index("Age"))

    # Download
    csv = df_preds.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload all three files to continue.")
