import streamlit as st
from streamlit_theme import st_theme


def hex_to_rgb(value):
    """Convert hex color (e.g., '#AABBCC') to (R, G, B) tuple."""
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def is_light_color(hex_color):
    """Determine if a color is light based on luminance."""
    r, g, b = hex_to_rgb(hex_color)
    # Calculate relative luminance (per W3C)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 128  # threshold (0–255 scale)


def set_page_config(app_icon_file_path="assets/PRIMA_app_icon.svg"):
    """Set Streamlit page configuration and sidebar logo based on theme."""
    st.set_page_config(
        page_title="PRIMA Online",
        page_icon=app_icon_file_path,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def remove_top_padding():
    """Remove top padding from Streamlit app."""
    st.markdown(
        """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def set_sidebar_logo(
    light_logo_path="assets/PRIMA_full_logo_v3.svg",
    dark_logo_path="assets/PRIMA_full_logo_v3_white.svg",
):
    """Set sidebar logo based on theme background color."""
    theme = st_theme()
    try:
        bg_color = theme.get("secondaryBackgroundColor", "#FFFFFF")
        if is_light_color(bg_color):
            st.sidebar.image(light_logo_path, width="stretch")
        else:
            st.sidebar.image(dark_logo_path, width="stretch")
    except Exception as e:
        print("Theme error:", e)
        st.sidebar.image(light_logo_path, width="stretch")
