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
model_choice = st.sidebar.selectbox("Choose a model", ["MAT", "BRT", "WA-PLS", "RF"])
target = st.sidebar.selectbox("Target climate variable", ["TANN", "Temp_season", "MTWA", "MTCO"])
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
    X_test = loader.load_test_data()
    X_train_aligned, X_test_aligned = loader.align_taxa(X_train, X_test)

    # Model selection
    if model_choice == "MAT":
        model = MAT(k=k)
    elif model_choice == "BRT":
        model = BRT(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=random_seed)
    elif model_choice == "WA-PLS":
        model = WA_PLS(n_components=pls_components)
    elif model_choice == "RF":
        model = RF(n_estimators=200, max_depth=10, random_state=random_seed)
    else:
        st.error("Invalid model choice")
        st.stop()

    # Train + predict
    model.fit(X_train_aligned, y_train)
    predictions = model.predict(X_test_aligned)

    # Display results
    st.subheader("Predictions")
    df_preds = pd.DataFrame({f"Predicted_{target}": predictions})
    st.write(df_preds)

    # Plot results
    st.subheader("Visualization")
    st.line_chart(df_preds)

    # Download option
    csv = df_preds.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload all three files to continue.")
