# tqdm_colors.py

# Define a fixed color map for models
color_map = {
    "MAT": "#1f77b4",  # blue
    "BRT": "#ff7f0e",  # orange
    "RF": "#2ca02c",  # green
}


class TQDMColors:
    """
    Centralized color codes for tqdm progress bars.
    Usage:
        from tqdm_colors import TQDMColors
        from tqdm import tqdm

        for _ in tqdm(range(100), bar_format=TQDMColors.GREEN + '{l_bar}{bar}{r_bar}' + TQDMColors.ENDC):
            ...
    """

    # ANSI color codes
    ENDC = "\033[0m"  # Reset color
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
