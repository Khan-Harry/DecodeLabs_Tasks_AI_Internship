# utils.py
"""Utility functions for the AI Recommendation System.

Provides dataset loading, overview display, missing value check, and feature info utilities.
"""

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV dataset.

    Args:
        path: Absolute path to ``dataset.csv``.
    Returns:
        pandas DataFrame with the data.
    """
    df = pd.read_csv(path)
    return df


def display_overview(df: pd.DataFrame) -> None:
    """Print basic information about the dataset.

    Shows number of records, column names, data types, and a preview of the first few rows.
    """
    print("--- Dataset Overview ---")
    print(f"Number of records: {len(df)}")
    print("Columns:")
    for col in df.columns:
        print(f" - {col}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    print("--- End Overview ---\n")


def check_missing(df: pd.DataFrame) -> None:
    """Report missing values per column.
    """
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values detected:")
        print(missing[missing > 0])
    else:
        print("No missing values detected.")


def get_feature_info(df: pd.DataFrame) -> None:
    """Print concise feature statistics.
    """
    print("--- Feature Statistics ---")
    if "Rating" in df.columns:
        print(f"Rating: min={df['Rating'].min()}, max={df['Rating'].max()}, mean={df['Rating'].mean():.2f}")
    if "Release Year" in df.columns:
        print(f"Release Year: min={df['Release Year'].min()}, max={df['Release Year'].max()}")
    if "Genre" in df.columns:
        print("Genre distribution:")
        print(df['Genre'].value_counts())
    print("--- End Statistics ---\n")
