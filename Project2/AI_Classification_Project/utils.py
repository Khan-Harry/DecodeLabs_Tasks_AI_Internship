# =============================================================================
# utils.py - Utility Helper Functions
# =============================================================================
# Description:
#   Provides shared utility functions used across the project:
#   - Console formatting (banners, separators, colored text)
#   - Input validation helpers
#   - Safe directory creation
#   - Timing utilities
# =============================================================================

import os
import time


# -----------------------------------------------------------------------------
# Console Formatting Helpers
# -----------------------------------------------------------------------------

def print_banner(title: str) -> None:
    """
    Print a formatted banner with the given title.

    Args:
        title (str): The title text to display inside the banner.
    """
    width = 70
    print("\n" + "=" * width)
    print(f"  {title.upper()}")
    print("=" * width)


def print_section(title: str) -> None:
    """
    Print a section header with dashes.

    Args:
        title (str): The section title text.
    """
    print(f"\n{'=' * 60}")
    print(f"  >>  {title}")
    print(f"{'=' * 60}")


def print_success(message: str) -> None:
    """Print a success message with a checkmark prefix."""
    print(f"  [OK]  {message}")


def print_info(message: str) -> None:
    """Print an informational message with an info prefix."""
    print(f"  [i]  {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"  [!]  WARNING: {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"  [X]  ERROR: {message}")


def print_separator() -> None:
    """Print a simple line separator."""
    print("-" * 60)


def print_menu_option(number: int, description: str) -> None:
    """
    Print a single menu option in a formatted style.

    Args:
        number  (int): The menu option number.
        description (str): Description of the menu option.
    """
    print(f"  [{number}]  {description}")


# -----------------------------------------------------------------------------
# Input Validation Helpers
# -----------------------------------------------------------------------------

def get_float_input(prompt: str, min_val: float = 0.0, max_val: float = 20.0) -> float:
    """
    Prompt the user for a valid float input within an allowed range.

    Args:
        prompt  (str):   The prompt text to display to the user.
        min_val (float): The minimum acceptable value (default 0.0).
        max_val (float): The maximum acceptable value (default 20.0).

    Returns:
        float: A validated float value entered by the user.
    """
    while True:
        try:
            value = float(input(prompt).strip())
            if min_val <= value <= max_val:
                return value
            else:
                print_warning(f"Value must be between {min_val} and {max_val}. Try again.")
        except ValueError:
            print_warning("Please enter a valid numeric value.")


def get_int_input(prompt: str, valid_choices: list) -> int:
    """
    Prompt the user for a valid integer choice from a list of valid options.

    Args:
        prompt        (str):  The prompt text.
        valid_choices (list): List of valid integer choices.

    Returns:
        int: A validated integer choice.
    """
    while True:
        try:
            choice = int(input(prompt).strip())
            if choice in valid_choices:
                return choice
            else:
                print_warning(f"Please enter one of: {valid_choices}")
        except ValueError:
            print_warning("Please enter a valid integer.")


def get_yes_no_input(prompt: str) -> bool:
    """
    Prompt the user for a yes/no answer.

    Args:
        prompt (str): The question to ask.

    Returns:
        bool: True for 'yes', False for 'no'.
    """
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        elif answer in ("n", "no"):
            return False
        else:
            print_warning("Please enter 'y' or 'n'.")


# -----------------------------------------------------------------------------
# File & Directory Helpers
# -----------------------------------------------------------------------------

def ensure_directory(path: str) -> None:
    """
    Create a directory (and any parent directories) if it does not exist.

    Args:
        path (str): The directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def get_project_root() -> str:
    """
    Return the absolute path of the project root directory.

    Returns:
        str: Absolute path to the directory containing this script.
    """
    return os.path.dirname(os.path.abspath(__file__))


def get_models_dir() -> str:
    """Return the absolute path to the 'models/' save directory."""
    path = os.path.join(get_project_root(), "models")
    ensure_directory(path)
    return path


def get_plots_dir() -> str:
    """Return the absolute path to the 'plots/' save directory."""
    path = os.path.join(get_project_root(), "plots")
    ensure_directory(path)
    return path


def get_outputs_dir() -> str:
    """Return the absolute path to the 'outputs/' save directory."""
    path = os.path.join(get_project_root(), "outputs")
    ensure_directory(path)
    return path


# -----------------------------------------------------------------------------
# Timing Utilities
# -----------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds into a human-readable string.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Human-readable duration string (e.g., '12.34 ms' or '1.23 s').
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.4f} s"
