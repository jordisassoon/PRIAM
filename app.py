import streamlit as st
from tabs import predictions, data_exploration, validation

st.set_page_config(page_title="Pollen Climate App", layout="wide")

st.sidebar.title("🌿 Pollen-based Climate Reconstruction")

# --- Sidebar: Shared Inputs ---
model_choice = st.sidebar.selectbox("Choose model", ["MAT", "WAPLS", "BRT", "RF", "All"])
target = st.sidebar.selectbox(
    "Target climate variable",
    ["TANN","Temp_season","MTWA","MTCO","PANN","Temp_wet","Temp_dry","P_wet","P_dry","P_season"]
)
n_neighbors = st.sidebar.slider("MAT neighbors", 1, 20, 5)
brt_trees = st.sidebar.slider("BRT trees", 1, 1000, 200)
rf_trees = st.sidebar.slider("RF trees", 1, 1000, 200)
cv_folds = st.sidebar.slider("CV folds", 1, 10, 4)
random_seed = st.sidebar.number_input("Random seed", value=42)

# File uploads
st.sidebar.header("Upload Data")
train_climate_file = st.sidebar.file_uploader("Training Climate CSV", type=["csv"])
train_pollen_file = st.sidebar.file_uploader("Training Pollen CSV", type=["csv"])
test_pollen_file = st.sidebar.file_uploader("Test Fossil Pollen CSV", type=["csv"])
taxa_mask_file = st.sidebar.file_uploader("Taxa mask CSV", type=["csv"])
coords_file = st.sidebar.file_uploader("Coordinates file (CSV)", type=["csv"])

# --- Tab selection ---
tab_choice = st.sidebar.radio("Select tab", ["Predictions", "Data Exploration", "Validation"])

# --- Render selected tab ---
if tab_choice == "Predictions":
    predictions.show_tab(
        train_climate_file, train_pollen_file, test_pollen_file,
        taxa_mask_file, model_choice, target, n_neighbors, brt_trees, rf_trees, 1, random_seed
    )
elif tab_choice == "Data Exploration":
    data_exploration.show_tab(train_climate_file, train_pollen_file, test_pollen_file, coords_file)
elif tab_choice == "Validation":
    validation.show_tab(
        train_climate_file, train_pollen_file, test_pollen_file,
        taxa_mask_file, model_choice, target, n_neighbors, brt_trees, rf_trees, cv_folds, random_seed
    )