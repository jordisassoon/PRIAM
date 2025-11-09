import streamlit as st
import yaml
import io
from utils.defaults import get_default_state_config


def initialize_state():
    """Initialize Streamlit session state with default values."""
    defaults = get_default_state_config()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state():
    """Reset Streamlit session state to default values."""
    defaults = get_default_state_config()
    for key in defaults.keys():
        st.session_state[key] = defaults[key]


def load_state_from_dict(state_dict: dict):
    """Load session state from a provided dictionary."""
    for key, value in state_dict.items():
        if "file" in key and not state_dict.get("use_dummy", False) and value is not None:
            st.sidebar.warning(f"Please use the following data for '{key}': {value}")
        elif key in get_default_state_config():
            st.session_state[key] = value
        else:
            st.warning(f"Unknown configuration key: {key}")


def load_state_from_yaml(uploaded_state: io.StringIO):
    """Load session state from a YAML string."""

    if uploaded_state is None:
        return

    if "uploaded_state" in st.session_state:
        if st.session_state["uploaded_state"] == uploaded_state.name:
            return  # Already loaded this state

    try:
        state_dict = yaml.safe_load(uploaded_state)
        load_state_from_dict(state_dict)

        st.session_state["uploaded_state"] = uploaded_state.name
    except Exception as e:
        st.error(f"Failed to load state from YAML: {e}")


def update_state(key: str, value):
    """Update a specific key in the session state."""
    st.session_state[key] = value
