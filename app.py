import streamlit as st
from tabs import predictions, data_exploration, validation
from streamlit_theme import st_theme
from utils.csv_loader import read_csv_auto_delimiter
import io
import yaml
import pandas as pd

st.set_page_config(page_title="PRISM Online",
    page_icon="assets/PRIAM_app_icon.svg", layout="wide", initial_sidebar_state="expanded")

# Remove top padding
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def hex_to_rgb(value):
    """Convert hex color (e.g., '#AABBCC') to (R, G, B) tuple."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def is_light_color(hex_color):
    """Determine if a color is light based on luminance."""
    r, g, b = hex_to_rgb(hex_color)
    # Calculate relative luminance (per W3C)
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    return luminance > 128  # threshold (0–255 scale)

theme = st_theme()

try:
    bg_color = theme.get("secondaryBackgroundColor", "#FFFFFF")
    if is_light_color(bg_color):
        st.sidebar.image("assets/PRIAM_full_logo_v3.svg", use_container_width=True)
    else:
        st.sidebar.image("assets/PRIAM_full_logo_v3_white.svg", use_container_width=True)
except Exception as e:
    print("Theme error:", e)
    st.sidebar.image("assets/PRIAM_full_logo_v3.svg", use_container_width=True)

st.sidebar.header("Model Configuration")

# --- Initialize session state defaults ---
defaults = {
    "model_choice": "MAT",
    "target": "TANN",
    "n_neighbors": 5,
    "brt_trees": 200,
    "rf_trees": 200,
    "cv_folds": 3,
    "random_seed": 42,
    "prediction_axis": "Age"
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Track uploaded config file in session state
if "uploaded_config_file" not in st.session_state:
    st.session_state["uploaded_config_file"] = None

# --- Sidebar: Load Config ---
uploaded_config = st.sidebar.file_uploader("Load State", type=["yaml", "yml"])
if uploaded_config and uploaded_config != st.session_state["uploaded_config_file"]:
    try:
        loaded_config = yaml.safe_load(uploaded_config)
        for key in loaded_config.keys():
            if "file" in key:
                if loaded_config[key] is not None:
                    st.sidebar.warning(f"Please re-upload the file for '{key}': {loaded_config[key]}")
                else:
                    st.session_state[key] = None
            elif key in defaults:
                st.session_state[key] = loaded_config[key]
            else:
                st.sidebar.warning(f"Unknown configuration key: {key}")
        st.session_state["uploaded_config_file"] = uploaded_config
        st.sidebar.success("Configuration loaded and applied!")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration: {e}")

# --- Sidebar: Model Configuration ---
model_choice = st.sidebar.selectbox(
    "Choose Model", ["MAT", "BRT", "RF", "All"], key="model_choice"
)

# Target depends on uploaded climate file
target_options = st.session_state.get("target_cols", ['TANN', 'PANN', 'MTWA', 'MTCO'])
target = st.sidebar.selectbox(
    "Target Climate Variable", target_options, key="target"
)

n_neighbors = st.sidebar.slider("MAT Neighbors", 1, 20, key="n_neighbors")
brt_trees = st.sidebar.slider("BRT Trees", 1, 1000, key="brt_trees")
rf_trees = st.sidebar.slider("RF Trees", 1, 1000, key="rf_trees")
cv_folds = st.sidebar.slider("CV Folds", 1, 10, key="cv_folds")
random_seed = st.sidebar.number_input("Random Seed", value=st.session_state["random_seed"], key="random_seed")

# --- Toggle for Predictions Representation ---
prediction_axis = st.sidebar.radio(
    "Show Predictions By:", ["Age", "Depth"], key="prediction_axis"
)

st.sidebar.header("Upload Data Files")
train_climate_file = st.sidebar.file_uploader("Training Climate CSV", type=["csv"])
train_proxy_file = st.sidebar.file_uploader("Training Proxy CSV", type=["csv"])
test_proxy_file = st.sidebar.file_uploader("Test Fossil Proxy CSV", type=["csv"])
taxa_mask_file = st.sidebar.file_uploader("Taxa mask CSV", type=["csv"])
coords_file = st.sidebar.file_uploader("Coordinates file (CSV)", type=["csv"])

def load_dummy_file(path):
    """Load a CSV from disk and return as a file-like object"""
    df = read_csv_auto_delimiter(open(path, "r", encoding="latin1"))
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)  # reset pointer to start
    return buffer

# --- Determine target variables dynamically ---
def get_climate_columns(climate_file):
    """Return list of column names from a climate CSV file-like object."""
    if climate_file is None:
        return []
    try:
        # reset pointer in case it's a BytesIO
        if hasattr(climate_file, "seek"):
            climate_file.seek(0)
        df = read_csv_auto_delimiter(climate_file).drop(["OBSNAME"], axis=1, errors="ignore")
        if hasattr(climate_file, "seek"):
            climate_file.seek(0)
        return df.columns.tolist()
    except Exception as e:
        st.sidebar.error(f"Failed to read climate file columns: {e}")
        return []

# --- File uploads / Dummy Data Toggle ---
use_dummy = st.sidebar.checkbox("Use Dummy Data")

if use_dummy:
    try:
        train_climate_file = load_dummy_file("./data/synthetic_climate_data.csv")
        train_proxy_file = load_dummy_file("./data/synthetic_modern_data.csv")
        test_proxy_file = load_dummy_file("./data/synthetic_test_data.csv")
        taxa_mask_file = None  # or load a dummy mask if available
        coords_file = load_dummy_file("./data/synthetic_coords_data.csv")
        st.sidebar.success("Dummy data loaded from ./data")
    except Exception as e:
        st.sidebar.error(f"Failed to load dummy data: {e}")
        train_climate_file = train_proxy_file = test_proxy_file = taxa_mask_file = coords_file = None

target_options = get_climate_columns(train_climate_file)

# only update state if file is present and has columns
if train_climate_file is not None and target_options:
    st.session_state["target_cols"] = target_options

# --- Toolbar: Save / Load Config ---
state = {
    "model_choice": model_choice,
    "target": target,
    "n_neighbors": n_neighbors,
    "brt_trees": brt_trees,
    "rf_trees": rf_trees,
    "cv_folds": cv_folds,
    "random_seed": random_seed,
    "prediction_axis": prediction_axis,
    # Include file names (or None if not uploaded)
    "train_climate_file": train_climate_file.name if train_climate_file and not use_dummy else None,
    "train_proxy_file": train_proxy_file.name if train_proxy_file and not use_dummy else None,
    "test_proxy_file": test_proxy_file.name if test_proxy_file and not use_dummy else None,
    "taxa_mask_file": taxa_mask_file.name if taxa_mask_file and not use_dummy else None,
    "coords_file": coords_file.name if coords_file and not use_dummy else None
}

st.sidebar.download_button(
    label="Save State",
    data=yaml.dump(state),
    file_name="priam_state.yaml",
    mime="text/yaml",
    use_container_width=True
)

# --- Tab Selection ---
tab_selection = st.segmented_control(
    "Select section:",
    options=["Predictions", "Data Exploration", "Validation"],
    default="Predictions",
    selection_mode="single",
    label_visibility="collapsed"
)

if tab_selection == "Predictions":
    predictions.show_tab(
        train_climate_file, train_proxy_file, test_proxy_file,
        taxa_mask_file, model_choice, target, n_neighbors,
        brt_trees, rf_trees, 1, random_seed, axis=prediction_axis
    )

elif tab_selection == "Data Exploration":
    data_exploration.show_tab(
        train_climate_file, train_proxy_file, test_proxy_file, coords_file,
        axis=prediction_axis
    )

elif tab_selection == "Validation":
    validation.show_tab(
        train_climate_file, train_proxy_file, test_proxy_file,
        taxa_mask_file, model_choice, target, n_neighbors,
        brt_trees, rf_trees, cv_folds, random_seed
    )
