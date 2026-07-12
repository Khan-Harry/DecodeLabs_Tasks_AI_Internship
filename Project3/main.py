# main.py
"""Entry point for the AI Recommendation System.

Provides an interactive command‑line menu that allows the user to:
* Load and explore the dataset
* Enter preferences
* Get rule‑based or similarity‑based recommendations
* Search for a movie by name
* Get recommendations similar to a chosen movie
* Rate recommended items
* View favorites and recommendation history (bonus)
* Visualize data (bonus)
* Export recommendations to CSV
* Exit the application
"""

import os
import sys
from typing import Dict, List, Tuple

import pandas as pd

# Local modules
from utils import load_dataset, display_overview, check_missing, get_feature_info
from user import collect_preferences, rate_item, add_favorite, view_favorites, view_history, user_ratings, recommendation_history, favorites
from recommender import rule_based_recommend, similarity_based_recommend, explain_recommendation
from similarity import build_item_matrix
from visualization import (
    plot_genre_distribution,
    plot_rating_distribution,
    plot_top_recommendations,
)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
EXPORT_PATH = os.path.join(os.path.dirname(__file__), "recommendations.csv")


def press_enter():
    input("\nPress Enter to continue...")


def display_menu() -> None:
    print("\n=== AI Recommendation System Menu ===")
    print("1. Show dataset overview")
    print("2. Enter/Update preferences")
    print("3. Rule‑based recommendation")
    print("4. Similarity‑based recommendation")
    print("5. Search movie by name")
    print("6. Recommend similar movies (by name)")
    print("7. View favorites")
    print("8. View recommendation history")
    print("9. Visualize data (genre / rating)")
    print("10. Export last recommendations to CSV")
    print("0. Exit")


def main() -> None:
    # Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        sys.exit(1)
    df = load_dataset(DATASET_PATH)
    # Pre‑compute TF‑IDF matrix for similarity recommendations
    vectorizer = build_item_matrix(df)

    # Store last recommendation list for export
    last_recs: List[Tuple[str, float]] = []
    # Preferences (initialize empty)
    prefs: Dict = {
        "genres": [],
        "language": "",
        "min_rating": 0.0,
        "keyword": "",
    }

    while True:
        display_menu()
        choice = input("Select an option: ").strip()
        if choice == "1":
            display_overview(df)
            check_missing(df)
            get_feature_info(df)
            press_enter()
        elif choice == "2":
            prefs = collect_preferences()
            print("Preferences updated:")
            print(prefs)
            press_enter()
        elif choice == "3":
            top = rule_based_recommend(df, prefs, top_n=5)
            if top.empty:
                print("No movies match your filters.")
            else:
                print("\n--- Rule‑Based Recommendations (Top 5) ---")
                for idx, (_, row) in enumerate(top.iterrows(), 1):
                    print(f"{idx}. {row['Movie Name']} (Genre: {row['Genre']}, Rating: {row['Rating']})")
                    print(explain_recommendation(row, prefs))
                    print("---")
                    # Record history
                    recommendation_history.append((row['Movie Name'], 0.0))
                    # Prompt for rating/favorite
                    rate_item(row['Movie Name'])
                    fav = input("Add to favorites? (y/n): ").strip().lower()
                    if fav == "y":
                        add_favorite(row['Movie Name'])
                last_recs = [(row['Movie Name'], 0.0) for _, row in top.iterrows()]
            press_enter()
        elif choice == "4":
            results = similarity_based_recommend(df, prefs, vectorizer, top_n=5)
            if not results:
                print("No movies matched after applying filters.")
            else:
                print("\n--- Similarity‑Based Recommendations (Top 5) ---")
                names = []
                scores = []
                for idx, (row, sim) in enumerate(results, 1):
                    names.append(row['Movie Name'])
                    scores.append(sim)
                    print(f"{idx}. {row['Movie Name']} (Genre: {row['Genre']}, Rating: {row['Rating']})")
                    print(f"Similarity: {sim * 100:.1f}%")
                    print(explain_recommendation(row, prefs, similarity=sim))
                    print("---")
                    recommendation_history.append((row['Movie Name'], sim))
                    rate_item(row['Movie Name'])
                    fav = input("Add to favorites? (y/n): ").strip().lower()
                    if fav == "y":
                        add_favorite(row['Movie Name'])
                last_recs = list(zip(names, scores))
                # Show bar chart
                plot_top_recommendations(names, scores)
            press_enter()
        elif choice == "5":
            name = input("Enter movie name to search: ").strip()
            matches = df[df["Movie Name"].str.lower() == name.lower()]
            if matches.empty:
                print("Movie not found.")
            else:
                row = matches.iloc[0]
                print("\n--- Movie Details ---")
                for col in df.columns:
                    print(f"{col}: {row[col]}")
                print("--- End Details ---")
            press_enter()
        elif choice == "6":
            name = input("Enter a movie you liked: ").strip()
            base = df[df["Movie Name"].str.lower() == name.lower()]
            if base.empty:
                print("Movie not found.")
                press_enter()
                continue
            # Use its keywords as a new query
            query_keyword = base.iloc[0]["Keywords"]
            temp_prefs = prefs.copy()
            temp_prefs["keyword"] = query_keyword
            results = similarity_based_recommend(df, temp_prefs, vectorizer, top_n=5)
            print(f"\nBecause you liked '{name}', you may also like:")
            for idx, (row, _) in enumerate(results, 1):
                if row['Movie Name'].lower() == name.lower():
                    continue
                print(f"{idx}. {row['Movie Name']} (Genre: {row['Genre']}, Rating: {row['Rating']})")
            press_enter()
        elif choice == "7":
            view_favorites()
            press_enter()
        elif choice == "8":
            view_history()
            press_enter()
        elif choice == "9":
            print("1. Genre distribution\n2. Rating distribution")
            sub = input("Choose chart: ").strip()
            if sub == "1":
                plot_genre_distribution(df)
            elif sub == "2":
                plot_rating_distribution(df)
            else:
                print("Invalid choice.")
            press_enter()
        elif choice == "10":
            if not last_recs:
                print("No recommendations to export yet.")
            else:
                export_df = pd.DataFrame(last_recs, columns=["Movie Name", "Similarity"])
                export_df.to_csv(EXPORT_PATH, index=False)
                print(f"Exported recommendations to {EXPORT_PATH}")
            press_enter()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
