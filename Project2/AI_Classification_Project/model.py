# =============================================================================
# model.py - Model Training, Saving, and Loading
# =============================================================================
# Description:
#   Defines, trains, saves, and loads the three classification models:
#   - Decision Tree Classifier  (primary model — interpretable)
#   - Logistic Regression       (comparison — linear baseline)
#   - K-Nearest Neighbors (KNN) (comparison — instance-based)
#
#   Each training function:
#   - Accepts training data
#   - Fits the model
#   - Measures training time
#   - Saves the trained model to disk as a .pkl file using joblib
# =============================================================================

import os
import time
import joblib
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from utils import (
    print_banner, print_section, print_success,
    print_info, print_error, get_models_dir, format_duration
)

# -----------------------------------------------------------------------------
# Model Constants
# -----------------------------------------------------------------------------

MODEL_NAMES = {
    "decision_tree":        "Decision Tree Classifier",
    "logistic_regression":  "Logistic Regression",
    "knn":                  "K-Nearest Neighbors (KNN)",
}

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Individual Model Training Functions
# -----------------------------------------------------------------------------

def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                        max_depth: int = None) -> tuple:
    """
    Train a Decision Tree Classifier.

    WHY DECISION TREE?
    A Decision Tree learns a hierarchy of if-else rules directly from the
    data. It is naturally interpretable — you can visualise exactly which
    feature thresholds the model uses to reach a prediction. It also
    provides a feature_importances_ attribute showing how much each feature
    contributes to reducing impurity across all splits.

    Args:
        X_train   (ndarray): Scaled training features.
        y_train   (ndarray): Training labels.
        max_depth (int):     Maximum depth of the tree. None = grow until
                             leaf nodes are pure (may overfit on large data).

    Returns:
        tuple: (model, training_time_seconds)
            - model         (DecisionTreeClassifier): Fitted model.
            - training_time (float): Wall-clock training duration in seconds.
    """
    model = DecisionTreeClassifier(
        criterion="gini",       # Use Gini impurity to choose splits
        max_depth=max_depth,    # Unconstrained depth for Iris (small dataset)
        random_state=RANDOM_STATE
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    return model, training_time


def train_logistic_regression(X_train: np.ndarray,
                               y_train: np.ndarray) -> tuple:
    """
    Train a Logistic Regression classifier.

    WHY LOGISTIC REGRESSION?
    Despite the name, Logistic Regression is a classification algorithm. It
    models the probability that a sample belongs to each class using a
    sigmoid/softmax function applied to a linear combination of features.
    It serves as a fast, interpretable linear baseline.

    Args:
        X_train (ndarray): Scaled training features.
        y_train (ndarray): Training labels.

    Returns:
        tuple: (model, training_time_seconds)
    """
    model = LogisticRegression(
        max_iter=200,           # Allow enough iterations to converge
        random_state=RANDOM_STATE,
        solver="lbfgs"          # Efficient solver for multiclass problems
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    return model, training_time


def train_knn(X_train: np.ndarray, y_train: np.ndarray,
              n_neighbors: int = 5) -> tuple:
    """
    Train a K-Nearest Neighbors classifier.

    WHY KNN?
    KNN makes predictions by finding the K closest training samples to a
    new data point (using Euclidean distance) and taking a majority vote of
    their labels. It requires no explicit training phase (lazy learning) but
    is sensitive to feature scales — hence scaling is critical before KNN.

    Args:
        X_train    (ndarray): Scaled training features.
        y_train    (ndarray): Training labels.
        n_neighbors (int):    Number of neighbours to consider. Default 5.

    Returns:
        tuple: (model, training_time_seconds)
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="euclidean",
        weights="uniform"
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    return model, training_time


# -----------------------------------------------------------------------------
# Train All Models at Once
# -----------------------------------------------------------------------------

def train_all_models(X_train: np.ndarray, y_train: np.ndarray,
                     verbose: bool = True) -> dict:
    """
    Train all three classification models and return them in a dictionary.

    Args:
        X_train (ndarray): Scaled training features.
        y_train (ndarray): Training labels.
        verbose (bool):    If True, print training progress and timings.

    Returns:
        dict: Keys are model short-names; values are (model, training_time) tuples.
              Example:
              {
                  "decision_tree":       (DecisionTreeClassifier, 0.002),
                  "logistic_regression": (LogisticRegression,     0.008),
                  "knn":                 (KNeighborsClassifier,   0.001),
              }
    """
    if verbose:
        print_banner("Model Training")

    results = {}

    # -- Decision Tree --------------------------------------------------------
    if verbose:
        print_section("Training Decision Tree Classifier")
    dt_model, dt_time = train_decision_tree(X_train, y_train)
    results["decision_tree"] = (dt_model, dt_time)
    if verbose:
        print_success(f"Decision Tree trained in {format_duration(dt_time)}")

    # -- Logistic Regression --------------------------------------------------
    if verbose:
        print_section("Training Logistic Regression")
    lr_model, lr_time = train_logistic_regression(X_train, y_train)
    results["logistic_regression"] = (lr_model, lr_time)
    if verbose:
        print_success(f"Logistic Regression trained in {format_duration(lr_time)}")

    # -- KNN ------------------------------------------------------------------
    if verbose:
        print_section("Training K-Nearest Neighbors (KNN)")
    knn_model, knn_time = train_knn(X_train, y_train)
    results["knn"] = (knn_model, knn_time)
    if verbose:
        print_success(f"KNN trained in {format_duration(knn_time)}")

    if verbose:
        print_section("Training Complete")
        print_success("All three models have been trained successfully.")

    return results


# -----------------------------------------------------------------------------
# Save & Load Models
# -----------------------------------------------------------------------------

def save_model(model, model_key: str) -> str:
    """
    Save a trained model to disk using joblib.

    Joblib serialises the Python object (model state, parameters, internal
    arrays) to a binary .pkl file. This lets us reload the trained model in
    a future session without retraining.

    Args:
        model     (estimator): The trained Scikit-Learn model object.
        model_key (str):       Short name key (e.g., 'decision_tree').

    Returns:
        str: Absolute path to the saved .pkl file.
    """
    models_dir = get_models_dir()
    filepath = os.path.join(models_dir, f"{model_key}.pkl")

    try:
        joblib.dump(model, filepath)
        print_success(f"Model saved -> {filepath}")
        return filepath
    except Exception as e:
        print_error(f"Failed to save model '{model_key}': {e}")
        raise


def load_model(model_key: str):
    """
    Load a previously saved model from disk.

    Args:
        model_key (str): Short name key (e.g., 'decision_tree').

    Returns:
        Loaded Scikit-Learn estimator, or None if the file does not exist.
    """
    models_dir = get_models_dir()
    filepath = os.path.join(models_dir, f"{model_key}.pkl")

    if not os.path.exists(filepath):
        print_error(f"No saved model found at: {filepath}")
        return None

    try:
        model = joblib.load(filepath)
        print_success(f"Model loaded <- {filepath}")
        return model
    except Exception as e:
        print_error(f"Failed to load model '{model_key}': {e}")
        raise


def save_all_models(trained_models: dict) -> None:
    """
    Save all trained models and the scaler to disk.

    Args:
        trained_models (dict): Dict from train_all_models() containing
                               (model, training_time) tuples.
    """
    print_section("Saving Models to Disk")
    for key, (model, _) in trained_models.items():
        save_model(model, key)


def load_all_models() -> dict:
    """
    Attempt to load all saved models from disk.

    Returns:
        dict: Keys are model short-names; values are loaded model objects
              (or None if not found).
    """
    loaded = {}
    for key in MODEL_NAMES:
        loaded[key] = load_model(key)
    return loaded


def save_scaler(scaler) -> str:
    """
    Save the fitted StandardScaler to disk.

    Args:
        scaler (StandardScaler): The fitted scaler from preprocessing.

    Returns:
        str: Absolute path of the saved scaler file.
    """
    models_dir = get_models_dir()
    filepath = os.path.join(models_dir, "scaler.pkl")
    try:
        joblib.dump(scaler, filepath)
        print_success(f"Scaler saved -> {filepath}")
        return filepath
    except Exception as e:
        print_error(f"Failed to save scaler: {e}")
        raise


def load_scaler():
    """
    Load the fitted StandardScaler from disk.

    Returns:
        StandardScaler, or None if not found.
    """
    models_dir = get_models_dir()
    filepath = os.path.join(models_dir, "scaler.pkl")
    if not os.path.exists(filepath):
        print_error(f"No saved scaler found at: {filepath}")
        return None
    try:
        scaler = joblib.load(filepath)
        print_success(f"Scaler loaded <- {filepath}")
        return scaler
    except Exception as e:
        print_error(f"Failed to load scaler: {e}")
        raise
