# visualization.py
"""Visualization utilities using Matplotlib.

Functions generate charts for genre distribution, rating distribution, top recommendations,
and similarity scores.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_genre_distribution(df: pd.DataFrame) -> None:
    """Bar chart of movie count per genre."""
    genre_counts = df["Genre"].value_counts()
    genre_counts.plot(kind="bar", color="steelblue")
    plt.title("Genre Distribution")
    plt.xlabel("Genre")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.show()


def plot_rating_distribution(df: pd.DataFrame) -> None:
    """Histogram of movie ratings."""
    plt.hist(df["Rating"], bins=10, color="seagreen", edgecolor="black")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_top_recommendations(names: list, scores: list) -> None:
    """Horizontal bar chart of top recommended movies with similarity scores.

    ``names`` – list of movie titles.
    ``scores`` – list of similarity scores (0‑1 range).
    """
    plt.barh(names, scores, color="mediumpurple")
    plt.xlabel("Similarity Score")
    plt.title("Top Recommendations")
    plt.gca().invert_yaxis()  # highest at top
    plt.tight_layout()
    plt.show()


def plot_similarity_bar(names: list, scores: list) -> None:
    """Alias for ``plot_top_recommendations`` – kept for semantic clarity."""
    plot_top_recommendations(names, scores)
