import streamlit as st
from tabs import predictions, data_exploration, validation
from streamlit_theme import st_theme

st.set_page_config(page_title="Pollen Climate App", layout="wide", initial_sidebar_state="expanded")

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
print(theme)

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

model_choice = st.sidebar.selectbox("Choose model", ["MAT", "WAPLS", "BRT", "RF", "All"])
target = st.sidebar.selectbox(
    "Target climate variable",
    ["TANN","Temp_season","MTWA","MTCO","PANN","Temp_wet","Temp_dry","P_wet","P_dry","P_season"]
)
n_neighbors = st.sidebar.slider("MAT neighbors", 1, 20, 5)
brt_trees = st.sidebar.slider("BRT trees", 1, 1000, 200)
rf_trees = st.sidebar.slider("RF trees", 1, 1000, 200)
cv_folds = st.sidebar.slider("CV folds", 1, 10, 5)
random_seed = st.sidebar.number_input("Random seed", value=42)

# --- File uploads ---
st.sidebar.header("Upload Data")
train_climate_file = st.sidebar.file_uploader("Training Climate CSV", type=["csv"])
train_pollen_file = st.sidebar.file_uploader("Training Pollen CSV", type=["csv"])
test_pollen_file = st.sidebar.file_uploader("Test Fossil Pollen CSV", type=["csv"])
taxa_mask_file = st.sidebar.file_uploader("Taxa mask CSV", type=["csv"])
coords_file = st.sidebar.file_uploader("Coordinates file (CSV)", type=["csv"])

# --- Top Tabs ---
tabs = st.tabs(["🔮 Predictions", "📊 Data Exploration", "✅ Validation"])

with tabs[0]:
    predictions.show_tab(
        train_climate_file, train_pollen_file, test_pollen_file,
        taxa_mask_file, model_choice, target, n_neighbors, brt_trees, rf_trees, 1, random_seed
    )

with tabs[1]:
    data_exploration.show_tab(
        train_climate_file, train_pollen_file, test_pollen_file, coords_file
    )

with tabs[2]:
    validation.show_tab(
        train_climate_file, train_pollen_file, test_pollen_file,
        taxa_mask_file, model_choice, target, n_neighbors, brt_trees, rf_trees,
        cv_folds, random_seed
    )
