import streamlit as st
from tabs import predictions, data_exploration, validation
from streamlit_theme import st_theme
from utils.csv_loader import read_csv_auto_delimiter
import io
import pandas as pd

st.set_page_config(page_title="PRISM Online",
    page_icon="assets/PRISM_app_icon.svg", layout="wide", initial_sidebar_state="expanded")

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
        st.sidebar.image("assets/PRISM_full_logo.svg", use_container_width=True)
    else:
        st.sidebar.image("assets/PRISM_full_logo_white.svg", use_container_width=True)
except Exception as e:
    print("Theme error:", e)
    st.sidebar.image("assets/PRISM_full_logo.svg", use_container_width=True)

# --- Shared Inputs (in Sidebar) ---
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose model", ["MAT", "BRT", "RF", "All"])

# selectbox reads from state
target = st.sidebar.selectbox(
    "Target climate variable",
    st.session_state.get("target_cols", ['TANN', 'PANN', 'MTWA', 'MTCO'])
)

n_neighbors = st.sidebar.slider("MAT neighbors", 1, 20, 5)
brt_trees = st.sidebar.slider("BRT trees", 1, 1000, 200)
rf_trees = st.sidebar.slider("RF trees", 1, 1000, 200)
cv_folds = st.sidebar.slider("CV folds", 1, 10, 3)
random_seed = st.sidebar.number_input("Random seed", value=42)

# --- Toggle for Predictions Representation ---
st.sidebar.header("Prediction Representation")
prediction_axis = st.sidebar.radio(
    "Show predictions by:",
    ["Age", "Depth"]
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
st.sidebar.header("Use Dummy Data")
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

tab_selection = st.segmented_control(
    "Select section:",  # still required, but hidden
    options=["Predictions", "Data Exploration", "Validation"],
    default="Predictions",
    selection_mode="single",
    label_visibility="collapsed"  # 👈 hides the label from the UI
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
