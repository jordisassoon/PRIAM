import streamlit as st
from tabs import predictions, data_exploration, validation
from streamlit_theme import st_theme

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
target = st.sidebar.selectbox(
    "Target climate variable",
    ["TANN","Temp_season","MTWA","MTCO","PANN","Temp_wet","Temp_dry","P_wet","P_dry","P_season"]
)
n_neighbors = st.sidebar.slider("MAT neighbors", 1, 20, 5)
brt_trees = st.sidebar.slider("BRT trees", 1, 1000, 200)
rf_trees = st.sidebar.slider("RF trees", 1, 1000, 200)
cv_folds = st.sidebar.slider("CV folds", 1, 10, 5)
random_seed = st.sidebar.number_input("Random seed", value=42)

# --- Toggle for Predictions Representation ---
st.sidebar.header("Prediction Representation")
prediction_axis = st.sidebar.radio(
    "Show predictions by:",
    ["Age", "Depth"]
)

# --- File uploads ---
st.sidebar.header("Upload Data")
train_climate_file = st.sidebar.file_uploader("Training Climate CSV", type=["csv"])
train_proxy_file = st.sidebar.file_uploader("Training Proxy CSV", type=["csv"])
test_proxy_file = st.sidebar.file_uploader("Test Fossil Proxy CSV", type=["csv"])
taxa_mask_file = st.sidebar.file_uploader("Taxa mask CSV", type=["csv"])
coords_file = st.sidebar.file_uploader("Coordinates file (CSV)", type=["csv"])

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
