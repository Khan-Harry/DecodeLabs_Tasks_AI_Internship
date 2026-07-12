# similarity.py
"""Similarity utilities using TF‑IDF and cosine similarity.

Provides functions to build a TF‑IDF matrix for the item keywords and to compute a
user preference vector based on the entered keyword(s).
"""

from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_item_matrix(df: pd.DataFrame) -> TfidfVectorizer:
    """Fit a TF‑IDF vectorizer on the ``Keywords`` column and return it.

    The function also adds a ``tfidf_matrix`` attribute to the DataFrame for
    later similarity queries.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    # Fill missing keywords with empty string
    tfidf_matrix = vectorizer.fit_transform(df["Keywords"].fillna(""))
    df["_tfidf_matrix"] = list(tfidf_matrix)  # store as list for easy access
    return vectorizer


def build_user_vector(keyword: str, vectorizer: TfidfVectorizer):
    """Transform a single keyword (or phrase) into the TF‑IDF space.
    """
    if not keyword:
        # Return a zero vector of correct shape
        import numpy as np
        return np.zeros((1, len(vectorizer.get_feature_names_out())))
    return vectorizer.transform([keyword])


def compute_similarity(user_vec, df: pd.DataFrame) -> List[float]:
    """Compute cosine similarity between the user vector and each item.

    Returns a list of similarity scores (float) aligned with ``df`` rows.
    """
    # Retrieve the TF‑IDF matrix stored in the DataFrame
    import numpy as np
    tfidf_matrix = pd.DataFrame(df["_tfidf_matrix"].tolist())
    # ``tfidf_matrix`` is a DataFrame of sparse rows; convert to scipy matrix
    tfidf_sparse = tfidf_matrix.iloc[:, 0].values[0] if tfidf_matrix.shape[1] == 1 else None
    # Simpler: use original vectorizer output (we saved it as a list of sparse rows)
    # Rebuild the matrix from the list
    from scipy.sparse import vstack
    item_matrix = vstack(df["_tfidf_matrix"].values)
    sims = cosine_similarity(user_vec, item_matrix).flatten()
    return sims.tolist()
