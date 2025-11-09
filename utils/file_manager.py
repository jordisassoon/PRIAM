import streamlit as st
import io
from utils.csv_loader import read_csv_auto_delimiter
from utils.state_manager import update_state


def load_dummy_file(path):
    """Load a CSV from disk and return as a file-like object"""
    df = read_csv_auto_delimiter(open(path, "r", encoding="latin1"))
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)  # reset pointer to start
    return buffer


# --- Determine target variables dynamically ---
def get_non_obs_columns(file):
    """Return list of column names from a climate CSV file-like object."""
    if file is None:
        return []
    try:
        # reset pointer in case it's a BytesIO
        if hasattr(file, "seek"):
            file.seek(0)
        df = read_csv_auto_delimiter(file).drop(["OBSNAME"], axis=1, errors="ignore")
        if hasattr(file, "seek"):
            file.seek(0)
        return df.columns.tolist()
    except Exception as e:
        st.sidebar.error(f"Failed to read climate file columns: {e}")
        return []


def load_file(uploaded_file):
    """Load an uploaded file into session state and return DataFrame."""
    if uploaded_file is None:
        return None
    try:
        # reset pointer in case it's a BytesIO
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        df = read_csv_auto_delimiter(uploaded_file)
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return df
    except Exception as e:
        st.error(f"Failed to load uploaded file: {e}")
        return None
