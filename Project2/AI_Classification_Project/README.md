# =============================================================================
# README.md — Data Classification Using Artificial Intelligence
# =============================================================================

# Data Classification Using Artificial Intelligence

> **A complete, professional, beginner-friendly supervised machine learning
> classification project built with Python and Scikit-Learn.**

---

## 📌 Project Description

This project demonstrates the **end-to-end supervised machine learning
pipeline** for classifying Iris flower species using three popular
classification algorithms. It is designed as an **internship portfolio
project** that showcases clean, modular, well-commented Python code alongside
professional data science practices.

---

## 🎯 Objective

Build a complete AI pipeline that:

1. **Loads** the Iris dataset from Scikit-Learn.
2. **Explores** the data through detailed EDA (statistics, distributions,
   correlations).
3. **Preprocesses** the data (train/test split, feature scaling).
4. **Trains** three classification models: Decision Tree, Logistic Regression,
   and K-Nearest Neighbors.
5. **Evaluates** model performance using accuracy, confusion matrix, and
   classification reports.
6. **Compares** models and identifies the best performer.
7. **Predicts** new, unseen flower species from user input.
8. **Visualises** results through multiple high-quality plots.

---

## 🛠️ Technologies

| Technology      | Purpose                                  |
|-----------------|------------------------------------------|
| Python 3        | Core programming language                |
| NumPy           | Numerical computations                   |
| Pandas          | Data manipulation and analysis           |
| Matplotlib      | Data visualisation and plotting          |
| Scikit-Learn    | Machine learning algorithms and tools    |
| Joblib          | Model serialisation (save / load)        |

---

## 🌸 Dataset Information

**Name:** Iris Dataset (Fisher's Iris)  
**Source:** `sklearn.datasets.load_iris()`  
**Samples:** 150 (50 per class)  
**Features:** 4 numerical features

| Feature        | Unit | Description                          |
|----------------|------|--------------------------------------|
| Sepal Length   | cm   | Length of the flower's sepal         |
| Sepal Width    | cm   | Width of the flower's sepal          |
| Petal Length   | cm   | Length of the flower's petal         |
| Petal Width    | cm   | Width of the flower's petal          |

**Target Classes:**

| Class Index | Species Name    |
|-------------|-----------------|
| 0           | Setosa          |
| 1           | Versicolor      |
| 2           | Virginica       |

---

## 🔄 Machine Learning Pipeline

```
Raw Data (Iris Dataset)
       │
       ▼
┌─────────────────┐
│  Data Loading   │  ← data_loader.py
│  & EDA          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │  ← preprocessing.py
│ Split + Scale   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │  ← model.py
│ DT | LR | KNN   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation     │  ← evaluation.py
│ Metrics+Compare │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualisation   │  ← visualization.py
│ Plots + Charts  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prediction     │  ← prediction.py
│  New Samples    │
└─────────────────┘
```

---

## 🤖 Algorithms Used

### 1. Decision Tree Classifier *(Primary Model)*
- Creates a hierarchy of if-else decision rules based on feature thresholds.
- **Interpretable**: the tree can be visualised and explained step-by-step.
- Provides `feature_importances_` showing which features drive predictions.

### 2. Logistic Regression *(Linear Baseline)*
- Models class probabilities using a softmax function applied to a linear
  combination of features.
- Fast, interpretable, and a standard benchmark for classification tasks.

### 3. K-Nearest Neighbors — KNN *(Instance-Based)*
- Predicts by finding the K closest training samples (Euclidean distance)
  and taking a majority vote.
- No explicit training phase (lazy learning); sensitive to feature scale.

---

## ⚙️ Installation Steps

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd AI_Classification_Project
```

### 2. (Recommended) Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Interactive CLI Menu

```bash
python main.py
```

You will see a numbered menu:

```
======================================================================
  DATA CLASSIFICATION USING ARTIFICIAL INTELLIGENCE
======================================================================

  [1]  Load & Explore Data
  [2]  Run Preprocessing
  [3]  Train All Models
  [4]  Evaluate & Compare Models
  [5]  Generate All Visualisations
  [6]  Prediction System
  [7]  Run Full Pipeline (Steps 1–5 Automatically)
  [0]  Exit
```

> **Tip:** Select **[7]** to run the entire pipeline automatically in one go.

---

## 📁 Project Structure

```
AI_Classification_Project/
│
├── main.py             # CLI menu system — project entry point
├── data_loader.py      # Iris dataset loading and EDA
├── preprocessing.py    # Train/test split and feature scaling
├── model.py            # Model training, saving, and loading
├── evaluation.py       # Metrics, comparison table, feature importance
├── visualization.py    # All Matplotlib plot generation
├── prediction.py       # Interactive prediction system (manual/batch/random)
├── utils.py            # Shared helper functions
│
├── requirements.txt    # Python package dependencies
├── README.md           # Project documentation (this file)
│
├── models/             # Saved .pkl model files (auto-created)
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   └── scaler.pkl
│
├── plots/              # Generated plot images (auto-created)
│   ├── 01_feature_histograms.png
│   ├── 02_scatter_petal.png
│   ├── 03_pair_plot.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_confusion_matrix_*.png
│   ├── 06_decision_tree.png
│   ├── 07_accuracy_comparison.png
│   └── 08_feature_importance.png
│
└── outputs/            # Exported CSV prediction files (auto-created)
    ├── predictions.csv
    └── test_set_predictions.csv
```

---

## 📊 Sample Output

### Model Comparison Table

```
  Algorithm                     Accuracy    Train Time
  ────────────────────────────────────────────────────
  Decision Tree Classifier         96.67%      1.23 ms
  Logistic Regression              100.00%     8.45 ms
  K-Nearest Neighbors (KNN)        100.00%     0.89 ms
  ────────────────────────────────────────────────────

  ✔  Best Performing Model: Logistic Regression
  ✔  Accuracy: 100.00%
```

### Decision Tree Feature Importances

```
  1. petal length (cm)             0.5765  |████████████████████████░░░░░░░░|
  2. petal width (cm)              0.3812  |███████████████████░░░░░░░░░░░░░|
  3. sepal length (cm)             0.0423  |██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
  4. sepal width (cm)              0.0000  |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
```

### Prediction Example

```
  INPUT FEATURES
  ────────────────────────────────────────────────────
    sepal length (cm)            :  5.10 cm
    sepal width (cm)             :  3.50 cm
    petal length (cm)            :  1.40 cm
    petal width (cm)             :  0.20 cm

  PREDICTION  [Decision Tree Classifier]
  ────────────────────────────────────────────────────
    🌸  Predicted Species : SETOSA
    📊  Confidence        : 100.00%
  ────────────────────────────────────────────────────
```

---

## 🔮 Future Improvements

1. **Cross-Validation** — Use k-fold cross-validation for more robust accuracy estimates.
2. **Hyperparameter Tuning** — Apply GridSearchCV or RandomizedSearchCV to optimise model parameters.
3. **More Algorithms** — Add Support Vector Machine (SVM), Random Forest, and Gradient Boosting.
4. **Web Interface** — Build a Flask or Streamlit web app for browser-based predictions.
5. **Additional Datasets** — Extend to other classification datasets (Wine, Breast Cancer, etc.).
6. **ROC Curves** — Plot ROC/AUC curves for a more detailed performance comparison.
7. **Learning Curves** — Show how model performance changes with training set size.
8. **Automated Reporting** — Generate a PDF report summarising the complete pipeline run.

---

## 📄 License

This project is developed for educational and internship portfolio purposes.  
Free to use, modify, and extend for learning.

---

*Built with ❤️ using Python and Scikit-Learn.*
