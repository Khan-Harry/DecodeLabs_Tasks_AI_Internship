import os
from datetime import datetime

def clean_input(user_input: str) -> str:
    """
    Cleans the user input by removing leading and trailing whitespace
    and converting the text to lowercase for consistent rule matching.
    
    Args:
        user_input (str): The raw string input from the user.
        
    Returns:
        str: The cleaned, normalized lowercase string.
    """
    if user_input is None:
        return ""
    return user_input.strip().lower()

def get_current_time() -> str:
    """
    Retrieves the current system time formatted as HH:MM:SS.
    
    Returns:
        str: The formatted current time.
    """
    return datetime.now().strftime("%H:%M:%S")

def get_current_date() -> str:
    """
    Retrieves today's date formatted as YYYY-MM-DD.
    
    Returns:
        str: The formatted current date.
    """
    return datetime.now().strftime("%Y-%m-%d")

def clear_screen() -> None:
    """
    Clears the console terminal screen in a platform-independent way.
    Runs 'cls' on Windows systems and 'clear' on Unix-based systems.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
