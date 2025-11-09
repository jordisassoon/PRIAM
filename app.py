import streamlit as st
import io
import yaml
import pandas as pd

from utils.page_loader import (
    set_page_config,
    set_sidebar_logo,
    remove_top_padding,
)
from utils.state_manager import (
    initialize_state,
    load_state_from_yaml,
    update_state,
)
from utils.file_manager import load_dummy_file, get_non_obs_columns, make_zip
from utils.defaults import get_default_state_config
from tabs import predictions, data_exploration, validation
from utils.csv_loader import read_csv_auto_delimiter
from utils.dataloader import ProxyDataLoader

# --- Set page config and sidebar logo ---
set_page_config()
set_sidebar_logo()
remove_top_padding()

# --- Initialize session state with defaults ---
initialize_state()

state_expander = st.sidebar.expander("Session Manager", expanded=False)
# --- Expander: Session Management ---
with state_expander:
    # --- Toolbar: Save / Load Config ---
    uploaded_state = st.file_uploader("Load Session", type=["yaml", "yml"])
    load_state_from_yaml(uploaded_state)

model_expander = st.sidebar.expander("Model Configuration", expanded=False)

with model_expander:
    # --- Sidebar: Model Configuration ---
    st.toggle("MAT", value=st.session_state["use_mat"], key="use_mat")
    st.toggle("BRT", value=st.session_state["use_brt"], key="use_brt")
    st.toggle("RF", value=st.session_state["use_rf"], key="use_rf")

    # Target depends on uploaded climate file
    target = st.selectbox(
        "Target Climate Variable",
        st.session_state.get("target_cols", []),
        key="target",
    )

    if st.session_state["use_mat"]:
        n_neighbors = st.number_input("MAT Neighbors", 1, 20, value=st.session_state["n_neighbors"], key="n_neighbors")
    if st.session_state["use_brt"]:
        brt_trees = st.number_input("BRT Trees", 1, 1000, value=st.session_state["brt_trees"], key="brt_trees")
        brt_learning_rate = st.number_input("BRT Learning Rate", 0.01, 1.0, value=st.session_state["brt_learning_rate"], key="brt_learning_rate")
        brt_max_depth = st.number_input("BRT Max Depth", 1, 20, value=st.session_state["brt_max_depth"], key="brt_max_depth")
    if st.session_state["use_rf"]:
        rf_trees = st.number_input("RF Trees", 1, 1000, value=st.session_state["rf_trees"], key="rf_trees")
        rf_max_depth = st.number_input("RF Max Depth", 1, 20, value=st.session_state["rf_max_depth"], key="rf_max_depth")
    cv_folds = st.number_input("CV Folds", 1, 10, value=st.session_state["cv_folds"], key="cv_folds")
    random_seed = st.number_input("Random Seed", value=st.session_state["random_seed"], key="random_seed")

    # --- Toggle for Predictions Representation ---
    prediction_axis = st.radio("Show Predictions By:", ["Age", "Depth"], key="prediction_axis")

    taxa_expander = st.expander("Select Taxa", expanded=False)
    with taxa_expander:
        if "taxa_cols" == {}:
            st.write("Taxa selection will appear here after uploading data files.")

data_expander = st.sidebar.expander("Data Loading", expanded=False)
with data_expander:
    # --- File uploads / Dummy Data Toggle ---
    use_dummy = st.checkbox(
        "Use Dummy Data",
        value=st.session_state.get("use_dummy", False),
        key="use_dummy",
    )

    if use_dummy:
        # --- Load dummy files ---
        train_climate_file = load_dummy_file("./data/synthetic_climate_data.csv")
        train_proxy_file = load_dummy_file("./data/synthetic_modern_data.csv")
        test_proxy_file = load_dummy_file("./data/synthetic_test_data.csv")
        taxa_mask_file = None
        coords_file = load_dummy_file("./data/synthetic_coords_data.csv")

        files = {
            "synthetic_climate_data.csv": train_climate_file,
            "synthetic_modern_data.csv": train_proxy_file,
            "synthetic_test_data.csv": test_proxy_file,
            "synthetic_coords_data.csv": coords_file,
        }

        # Create ZIP
        zip_buffer = make_zip(files, "dummy_data.zip")

        # Download button
        st.download_button(
            label="Download Dummy Data", data=zip_buffer, file_name="dummy_data.zip", mime="application/zip", use_container_width=True
        )
    else:
        # --- Sidebar: File Uploads ---
        train_climate_file = st.file_uploader("Training Climate CSV", type=["csv"])
        train_proxy_file = st.file_uploader("Training Proxy CSV", type=["csv"])
        test_proxy_file = st.file_uploader("Test Fossil Proxy CSV", type=["csv"])
        taxa_mask_file = st.file_uploader("Taxa mask CSV", type=["csv"])
        coords_file = st.file_uploader("Coordinates file (CSV)", type=["csv"])

        update_state("train_climate_file", train_climate_file.name if train_climate_file else None)
        update_state("train_proxy_file", train_proxy_file.name if train_proxy_file else None)
        update_state("test_proxy_file", test_proxy_file.name if test_proxy_file else None)
        update_state("taxa_mask_file", taxa_mask_file.name if taxa_mask_file else None)
        update_state("coords_file", coords_file.name if coords_file else None)

if train_climate_file is not None and train_proxy_file is not None and test_proxy_file is not None:
    # --- Load Data ---
    loader = ProxyDataLoader(
        climate_file=train_climate_file,
        proxy_file=train_proxy_file,
        test_file=test_proxy_file,
        mask_file=taxa_mask_file,
    )

    X_train, y_train, train_metadata = loader.load_training_data(target)
    X_test, test_metadata = loader.load_test_data(age_or_depth=prediction_axis)
    X_train_aligned, X_test_aligned, shared_cols = loader.align_taxa(X_train, X_test)

    # --- Taxa selection expander ---
    if shared_cols is not None and len(shared_cols) > 0:
        with model_expander:
            with taxa_expander:
                # Dictionary to hold user selections
                taxa_selection = {}
                for taxa in shared_cols:
                    if taxa in st.session_state.get("taxa_cols", {}).keys():
                        taxa_selection[taxa] = st.checkbox(
                            taxa,
                            value=st.session_state["taxa_cols"].get(taxa, False),
                        )  # default checked
                    else:
                        taxa_selection[taxa] = st.checkbox(taxa, value=True)  # default checked

                # Filter columns based on selections
                selected_taxa = [taxa for taxa, include in taxa_selection.items() if include]

                if len(selected_taxa) == 0:
                    st.warning("No taxa selected. Predictions will fail.")
                else:
                    # Apply selection to aligned data
                    X_train_aligned = X_train_aligned[selected_taxa]
                    X_test_aligned = X_test_aligned[selected_taxa]

                st.session_state["taxa_cols"] = {taxa: True for taxa in selected_taxa}  # Update selected taxa
                st.session_state["taxa_cols"].update({taxa: False for taxa in shared_cols if taxa not in selected_taxa})
    # TODO: update shared_cols
else:
    X_train_aligned = None
    X_test_aligned = None
    y_train = None
    train_metadata = None
    test_metadata = None
    loader = None

target_options = get_non_obs_columns(train_climate_file)
if train_climate_file is not None and target_options:
    st.session_state["target_cols"] = target_options

# --- Tab Selection ---
tab_selection = st.segmented_control(
    "Select section:",
    options=["Predictions", "Data Exploration", "Validation"],
    default="Predictions",
    selection_mode="single",
    label_visibility="collapsed",
)

if tab_selection == "Predictions":
    predictions.show_tab(
        X_train=X_train_aligned,
        X_test=X_test_aligned,
        y_train=y_train,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
    )

elif tab_selection == "Data Exploration":
    data_exploration.show_tab(
        train_climate_file,
        train_proxy_file,
        test_proxy_file,
        coords_file,
        axis=prediction_axis,
    )

elif tab_selection == "Validation":
    validation.show_tab(
        X_train=X_train_aligned,
        y_train=y_train,
        loader=loader,
        train_metadata=train_metadata,
    )

with state_expander:
    st.download_button(
        label="Save Session",
        data=yaml.dump(st.session_state.to_dict()),
        file_name="priam_session.yaml",
        mime="text/yaml",
        width="stretch",
    )
