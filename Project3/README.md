# AI Recommendation System

## Project Overview

This repository contains a **beginner‑friendly, professional‑grade recommendation system** built entirely in Python. The application demonstrates the full recommendation pipeline:

1. **Data loading** – a small CSV dataset of movies (50 records) with attributes such as genre, language, rating, duration, release year, and keywords.
2. **User preference collection** – interactive CLI collects favorite genres, language, minimum rating, and keyword(s).
3. **Rule‑based filtering** – straightforward matching on genre, language and rating.
4. **Similarity‑based recommendation** – TF‑IDF vectorisation of the keyword field and cosine similarity to rank items.
5. **Additional features** – search by name, recommend items similar to a chosen movie, rating of recommendations, favorites list, recommendation history, data visualisations, and CSV export.

The project is modular, follows PEP‑8, includes clear docstrings, and uses only the allowed libraries (pandas, numpy, scikit‑learn, matplotlib).

---
## Objective

Create a complete AI recommendation system that showcases traditional (non‑deep‑learning) recommendation techniques suitable for an AI internship portfolio.

---
## Technologies Used

- **Python 3**
- **pandas** – data manipulation and CSV handling
- **numpy** – numerical operations
- **scikit‑learn** – TF‑IDF vectoriser and cosine similarity
- **matplotlib** – visualisations (genre / rating distribution, recommendation scores)

---
## Dataset Description

`dataset.csv` – 50 movies with the following columns:

| Column | Description |
|--------|-------------|
| **Movie Name** | Title of the movie |
| **Genre** | Primary genre (e.g., Sci‑Fi, Action, Drama) |
| **Language** | Original language |
| **Rating** | IMDb‑style rating (0‑10) |
| **Duration** | Length in minutes |
| **Release Year** | Year the movie was released |
| **Keywords** | Space‑separated descriptive keywords used for similarity |

---
## Project Structure

```
AI_Recommendation_System/
│── main.py               # CLI entry point
│── dataset.csv           # Sample movie dataset
│── requirements.txt      # Python dependencies
│── utils.py              # Data loading & overview helpers
│── user.py               # Preference collection & session state
│── similarity.py         # TF‑IDF & cosine similarity utilities
│── recommender.py        # Rule‑based & similarity‑based logic
│── visualization.py      # Matplotlib chart functions
│── README.md             # This document
```

---
## Installation

1. **Clone the repository** (or copy the folder into your workspace).
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---
## How to Run

```bash
python main.py
```

You will be presented with an interactive menu. Typical workflow:

1. **Show dataset overview** – verify the data.
2. **Enter/Update preferences** – provide genre(s), language, minimum rating, and an optional keyword.
3. Choose either **Rule‑based** or **Similarity‑based** recommendation.
4. Review the top‑5 results, rate items, and optionally add them to your favorites.
5. Use the other menu options to search, view similar movies, visualise data, or export the last recommendation list to `recommendations.csv`.

---
## Features

- Rule‑based filtering (genre, language, rating)
- Keyword similarity using TF‑IDF + cosine similarity
- Search by exact movie name
- “Because you liked X, you may also like…” similar‑item recommendation
- In‑session rating system that influences future similarity scores (simple additive boost)
- Favorites list management
- Recommendation history view
- Data visualisations (genre distribution, rating histogram, recommendation scores)
- Export of recommendations to CSV
- Clean, modular code with extensive comments and docstrings

---
## Future Improvements

- Implement collaborative filtering or matrix factorisation for personalised recommendations.
- Add a simple user authentication system and persistent storage (SQLite or JSON).
- Develop a web front‑end (Flask/FastAPI) for a browser‑based UI.
- Incorporate additional content features (e.g., plot summaries) for richer TF‑IDF vectors.
- Use a hybrid approach combining rule‑based and similarity scores.
- Deploy the app to a cloud platform (Heroku, Render, etc.) with CI/CD.
- Real‑time recommendation updates as users rate items.

---
## License

This project is provided for educational purposes and may be freely used, modified, and shared.

---
## Acknowledgements

The dataset is a curated collection of popular movies created for this demo. The code follows best practices inspired by open‑source Python projects and the scikit‑learn documentation.
