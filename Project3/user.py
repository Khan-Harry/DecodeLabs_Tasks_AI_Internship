# user.py
"""User interaction utilities.

Collects preferences, validates input, and manages in‑session data such as ratings,
history, and favorites.
"""

from typing import Dict, List, Tuple


def _input_prompt(prompt: str) -> str:
    """Wrapper for ``input`` to allow easy mocking/testing."""
    return input(prompt).strip()


def collect_preferences() -> Dict[str, any]:
    """Prompt the user for their movie preferences.

    Returns a dictionary with keys: ``genre`` (list of str), ``language`` (str),
    ``min_rating`` (float), ``keyword`` (str). Empty inputs are treated as no
    preference.
    """
    print("--- Preference Collection ---")
    genre_input = _input_prompt("Favorite Genre(s) (comma‑separated, or leave blank): ")
    genres = [g.strip() for g in genre_input.split(",") if g.strip()] if genre_input else []

    language = _input_prompt("Preferred Language (or leave blank): ")
    min_rating_str = _input_prompt("Minimum Rating (0‑10, or leave blank): ")
    try:
        min_rating = float(min_rating_str) if min_rating_str else 0.0
    except ValueError:
        print("Invalid rating, defaulting to 0.")
        min_rating = 0.0

    keyword = _input_prompt("Keyword (or leave blank): ")
    prefs = {
        "genres": genres,
        "language": language.lower() if language else "",
        "min_rating": min_rating,
        "keyword": keyword.lower() if keyword else "",
    }
    return prefs


# Simple in‑memory stores for the session
user_ratings: Dict[str, float] = {}
recommendation_history: List[Tuple[str, float]] = []  # (movie name, similarity score)
favorites: List[str] = []


def rate_item(item_name: str) -> None:
    """Ask the user to rate an item (1‑5) and store the result."""
    while True:
        rating_str = _input_prompt(f"Rate '{item_name}' (1‑5, or 0 to skip): ")
        try:
            rating = int(rating_str)
            if rating == 0:
                return
            if 1 <= rating <= 5:
                user_ratings[item_name] = rating
                print(f"Recorded rating {rating} for {item_name}.")
                return
        except ValueError:
            pass
        print("Please enter a number between 1 and 5, or 0 to skip.")


def add_favorite(item_name: str) -> None:
    if item_name not in favorites:
        favorites.append(item_name)
        print(f"Added '{item_name}' to favorites.")
    else:
        print(f"'{item_name}' is already in favorites.")


def view_favorites() -> None:
    if not favorites:
        print("No favorites saved yet.")
    else:
        print("--- Favorites ---")
        for idx, name in enumerate(favorites, 1):
            print(f"{idx}. {name}")
        print("--- End Favorites ---")


def view_history() -> None:
    if not recommendation_history:
        print("No recommendation history yet.")
    else:
        print("--- Recommendation History ---")
        for idx, (name, score) in enumerate(recommendation_history, 1):
            print(f"{idx}. {name} (score: {score:.2%})")
        print("--- End History ---")
