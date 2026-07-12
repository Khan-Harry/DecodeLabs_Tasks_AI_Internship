# =============================================================================
# evaluation.py - Model Evaluation and Performance Metrics
# =============================================================================
# Description:
#   Handles all post-training evaluation tasks:
#   - Accuracy score
#   - Confusion matrix display
#   - Full classification report (precision, recall, F1-score)
#   - Formatted model comparison table
#   - Best model identification
#   - Decision Tree feature importances
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from model import MODEL_NAMES
from utils import (
    print_banner, print_section, print_success,
    print_info, print_separator
)


# -----------------------------------------------------------------------------
# Single-Model Evaluation
# -----------------------------------------------------------------------------

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   target_names: list, model_name: str = "Model") -> dict:
    """
    Evaluate a trained model on the test set and print all metrics.

    Metrics computed:
    - Accuracy   : Fraction of correctly classified samples.
    - Confusion Matrix : Table showing true vs predicted class counts.
      Rows = actual classes, columns = predicted classes.
      Diagonal entries = correct predictions; off-diagonal = errors.
    - Classification Report : Per-class precision, recall, F1-score.
      * Precision = TP / (TP + FP)  — how many predicted positives are correct?
      * Recall    = TP / (TP + FN)  — how many actual positives were found?
      * F1-score  = harmonic mean of precision and recall

    Args:
        model        (estimator): Trained Scikit-Learn classifier.
        X_test       (ndarray):   Scaled test features.
        y_test       (ndarray):   True test labels.
        target_names (list):      List of class names (e.g. ['setosa', ...]).
        model_name   (str):       Human-readable model label for display.

    Returns:
        dict: {
            'accuracy':     float,
            'confusion_matrix': ndarray,
            'y_pred':       ndarray,
        }
    """
    # -- Predictions ----------------------------------------------------------
    y_pred = model.predict(X_test)

    # -- Accuracy -------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)

    # -- Confusion Matrix -----------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    # -- Classification Report ------------------------------------------------
    report = classification_report(y_test, y_pred, target_names=target_names)

    # -- Display --------------------------------------------------------------
    print_banner(f"Evaluation: {model_name}")

    print_section("Accuracy Score")
    print_success(f"Accuracy: {acc * 100:.2f}%  ({int(acc * len(y_test))}/{len(y_test)} correct)")

    print_section("Confusion Matrix")
    _print_confusion_matrix(cm, target_names)

    print_section("Classification Report")
    print(report)

    return {
        "accuracy":        acc,
        "confusion_matrix": cm,
        "y_pred":          y_pred,
    }


def _print_confusion_matrix(cm: np.ndarray, target_names: list) -> None:
    """
    Pretty-print a confusion matrix with class labels.

    Args:
        cm           (ndarray): Confusion matrix from sklearn.
        target_names (list):    Class name labels.
    """
    # Header row
    header = f"{'':>15}" + "".join(f"  {name[:10]:>10}" for name in target_names)
    print(header)
    print_separator()

    for i, row_name in enumerate(target_names):
        row_str = f"  {row_name[:13]:>13}" + "".join(f"  {cm[i][j]:>10}" for j in range(len(target_names)))
        print(row_str)

    print()
    print_info("Rows = Actual class | Columns = Predicted class")
    print_info("Diagonal = Correct predictions | Off-diagonal = Misclassifications")


# -----------------------------------------------------------------------------
# Multi-Model Comparison
# -----------------------------------------------------------------------------

def compare_models(trained_models: dict, X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    """
    Evaluate all trained models and collect their accuracy scores.

    Args:
        trained_models (dict): Dict of {model_key: (model, training_time)}.
        X_test         (ndarray): Scaled test features.
        y_test         (ndarray): True test labels.

    Returns:
        dict: {model_key: accuracy_float} for every model.
    """
    accuracies = {}
    for key, (model, _) in trained_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[key] = acc
    return accuracies


def display_comparison_table(trained_models: dict, accuracies: dict) -> str:
    """
    Print a formatted table comparing model accuracies and training times,
    then identify and announce the best-performing model.

    Args:
        trained_models (dict): Dict of {model_key: (model, training_time)}.
        accuracies     (dict): Dict of {model_key: accuracy_float}.

    Returns:
        str: The model_key of the best-performing model.
    """
    print_banner("Model Comparison")

    # -- Table header ---------------------------------------------------------
    col_w = [28, 12, 14]
    header = (f"  {'Algorithm':<{col_w[0]}}"
              f"{'Accuracy':>{col_w[1]}}"
              f"{'Train Time':>{col_w[2]}}")
    divider = "  " + "-" * (sum(col_w) + 2)

    print()
    print(header)
    print(divider)

    best_key = max(accuracies, key=accuracies.get)

    for key, (model, train_time) in trained_models.items():
        from utils import format_duration
        acc = accuracies[key]
        marker = " <<< BEST" if key == best_key else ""
        row = (f"  {MODEL_NAMES[key]:<{col_w[0]}}"
               f"{acc * 100:>{col_w[1]}.2f}%"
               f"{format_duration(train_time):>{col_w[2]}}"
               f"{marker}")
        print(row)

    print(divider)
    print()
    print_success(f"Best Performing Model: {MODEL_NAMES[best_key]}")
    print_success(f"Accuracy: {accuracies[best_key] * 100:.2f}%")

    return best_key


# -----------------------------------------------------------------------------
# Feature Importance (Decision Tree)
# -----------------------------------------------------------------------------

def display_feature_importance(dt_model, feature_names: list) -> None:
    """
    Display sorted feature importances from the Decision Tree.

    Decision Trees compute feature importance as the weighted reduction in
    impurity (Gini) attributed to each feature across all splits. A higher
    value means the model relies on that feature more for its decisions.

    Args:
        dt_model      (DecisionTreeClassifier): Trained Decision Tree model.
        feature_names (list): List of feature name strings.
    """
    print_banner("Decision Tree Feature Importances")

    importances = dt_model.feature_importances_

    # Sort by importance descending
    sorted_indices = np.argsort(importances)[::-1]

    print_section("Importance Scores (higher = more influential)")
    print()

    for rank, idx in enumerate(sorted_indices, start=1):
        name = feature_names[idx]
        score = importances[idx]
        bar_len = int(score * 40)           # Scale to 40-char bar
        bar = "#" * bar_len + "-" * (40 - bar_len)
        print(f"  {rank}. {name:<30} {score:.4f}  |{bar}|")

    print()
    print_info("Petal-related features typically dominate for the Iris dataset,")
    print_info("as petal dimensions have the clearest class separation.")


# -----------------------------------------------------------------------------
# Full Evaluation Pipeline
# -----------------------------------------------------------------------------

def run_full_evaluation(trained_models: dict,
                        X_test: np.ndarray, y_test: np.ndarray,
                        feature_names: list, target_names: list) -> tuple:
    """
    Run complete evaluation: per-model metrics, comparison table, and
    Decision Tree feature importances.

    Args:
        trained_models (dict):  Dict from train_all_models().
        X_test         (ndarray): Scaled test features.
        y_test         (ndarray): True test labels.
        feature_names  (list):  Feature column names.
        target_names   (list):  Class name strings.

    Returns:
        tuple: (eval_results, accuracies, best_model_key)
            - eval_results  (dict): Per-model evaluation dictionaries.
            - accuracies    (dict): {model_key: accuracy} mapping.
            - best_model_key (str): Key of the best-performing model.
    """
    eval_results = {}

    # -- Evaluate each model --------------------------------------------------
    for key, (model, _) in trained_models.items():
        result = evaluate_model(
            model, X_test, y_test,
            target_names=target_names,
            model_name=MODEL_NAMES[key]
        )
        eval_results[key] = result

    # -- Comparison table -----------------------------------------------------
    accuracies = {key: res["accuracy"] for key, res in eval_results.items()}
    best_key = display_comparison_table(trained_models, accuracies)

    # -- Feature importances (Decision Tree only) ------------------------------
    if "decision_tree" in trained_models:
        dt_model = trained_models["decision_tree"][0]
        display_feature_importance(dt_model, feature_names)

    return eval_results, accuracies, best_key
