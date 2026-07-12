# recommender.py
"""Recommendation engine combining rule‑based filtering and similarity‑based ranking.

Exports:
* ``rule_based_recommend(df, prefs, top_n=5)`` – returns a list of matching rows.
* ``similarity_based_recommend(df, prefs, vectorizer, top_n=5)`` – returns top‑N items with similarity scores.
* ``explain_recommendation(item, prefs, similarity=None)`` – human‑readable explanation string.
"""

from typing import List, Tuple, Dict
import pandas as pd

from similarity import build_user_vector, compute_similarity


def rule_based_recommend(df: pd.DataFrame, prefs: Dict, top_n: int = 5) -> pd.DataFrame:
    """Filter the DataFrame based on genre, language and minimum rating.

    * ``prefs["genres"]`` – list of acceptable genres (empty list means any).
    * ``prefs["language"]`` – required language (empty string means any).
    * ``prefs["min_rating"]`` – numeric lower bound.
    Returns the first ``top_n`` rows after filtering.
    """
    filtered = df.copy()
    if prefs.get("genres"):
        filtered = filtered[filtered["Genre"].isin(prefs["genres"])]
    if prefs.get("language"):
        filtered = filtered[filtered["Language"].str.lower() == prefs["language"].lower()]
    if prefs.get("min_rating"):
        filtered = filtered[filtered["Rating"] >= prefs["min_rating"]]
    # Sort by rating descending for a nicer default ordering
    filtered = filtered.sort_values(by="Rating", ascending=False)
    return filtered.head(top_n)


def similarity_based_recommend(df: pd.DataFrame, prefs: Dict, vectorizer, top_n: int = 5) -> List[Tuple[pd.Series, float]]:
    """Compute cosine similarity between the user's keyword(s) and each item.

    The function also respects genre, language and rating filters – items that do not
    satisfy those hard constraints receive a similarity of ``0`` and are excluded.
    Returns a list of ``(row, similarity)`` tuples sorted descending.
    """
    # Build user TF‑IDF vector from the keyword preference
    user_vec = build_user_vector(prefs.get("keyword", ""), vectorizer)
    # Compute raw similarity for all items
    sims = compute_similarity(user_vec, df)
    df = df.copy()
    df["_sim"] = sims
    # Apply hard filters (same as rule‑based) – items failing filters get 0 similarity
    if prefs.get("genres"):
        df.loc[~df["Genre"].isin(prefs["genres"]), "_sim"] = 0.0
    if prefs.get("language"):
        df.loc[df["Language"].str.lower() != prefs["language"].lower(), "_sim"] = 0.0
    if prefs.get("min_rating"):
        df.loc[df["Rating"] < prefs["min_rating"], "_sim"] = 0.0
    # Sort and take top_n
    top = df.sort_values(by="_sim", ascending=False).head(top_n)
    results = [(row, row["_sim"]) for _, row in top.iterrows() if row["_sim"] > 0]
    return results


def explain_recommendation(item: pd.Series, prefs: Dict, similarity: float = None) -> str:
    """Generate a multi‑line explanation why *item* was recommended.
    """
    reasons = []
    # Genre match
    if not prefs.get("genres") or item["Genre"] in prefs["genres"]:
        reasons.append("✔ Genre matches")
    # Language match
    if not prefs.get("language") or item["Language"].lower() == prefs["language"].lower():
        reasons.append("✔ Language matches")
    # Rating constraint
    if not prefs.get("min_rating") or item["Rating"] >= prefs["min_rating"]:
        reasons.append("✔ Rating meets preference")
    # Keyword similarity
    if similarity is not None:
        percent = similarity * 100
        reasons.append(f"✔ Keyword similarity: {percent:.0f}%")
    return "\n".join(reasons)
