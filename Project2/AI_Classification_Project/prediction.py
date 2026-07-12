# =============================================================================
# prediction.py - Prediction System
# =============================================================================
# Description:
#   Provides an interactive prediction system with four modes:
#   1. Manual input   — user enters one sample's feature values
#   2. Batch input    — user enters multiple samples at once
#   3. Random sample  — randomly picks a sample from the test set and predicts
#   4. Export to CSV  — saves a batch of predictions to outputs/predictions.csv
#
#   Each prediction displays:
#   - Predicted species name
#   - Prediction confidence/probability (if the model supports predict_proba)
# =============================================================================

import os
import csv
import numpy as np
import pandas as pd

from utils import (
    print_banner, print_section, print_success,
    print_info, print_warning, print_error, print_separator,
    get_float_input, get_int_input, get_yes_no_input,
    get_outputs_dir, format_duration
)
import time


# -----------------------------------------------------------------------------
# Core Prediction Function
# -----------------------------------------------------------------------------

def predict_sample(model, scaler, features: np.ndarray,
                   target_names: list) -> tuple:
    """
    Predict the species for a single sample and return label + confidence.

    Args:
        model        (estimator): Trained Scikit-Learn classifier.
        scaler       (StandardScaler): Fitted scaler from preprocessing.
        features     (ndarray):  Raw (un-scaled) feature values, shape (4,).
        target_names (list):     Class name strings.

    Returns:
        tuple: (predicted_class_name, confidence_percent_or_None, class_index)
            - predicted_class_name (str):   Human-readable species name.
            - confidence           (float): Probability (0–100) or None.
            - class_index          (int):   Numeric class label.
    """
    # Reshape to 2D array: model expects shape (1, 4)
    sample = features.reshape(1, -1)

    # Scale the raw input using the fitted scaler
    sample_scaled = scaler.transform(sample)

    # -- Prediction ------------------------------------------------------------
    class_index = model.predict(sample_scaled)[0]
    predicted_name = target_names[class_index]

    # -- Confidence (probability) ---------------------------------------------
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_scaled)[0]
        confidence = proba[class_index] * 100.0

    return predicted_name, confidence, class_index


def _display_prediction_result(features: np.ndarray,
                                feature_names: list,
                                predicted_name: str,
                                confidence,
                                target_names: list,
                                model_name: str = "Model") -> None:
    """
    Print a formatted prediction result to the console.

    Args:
        features       (ndarray): Raw feature values.
        feature_names  (list):    Feature column names.
        predicted_name (str):     Predicted class label.
        confidence     (float):   Probability percentage (or None).
        target_names   (list):    All class names (to display full proba bar).
        model_name     (str):     Name of the model used.
    """
    print()
    print_separator()
    print(f"  INPUT FEATURES")
    print_separator()
    for name, val in zip(feature_names, features):
        print(f"    {name:<30}: {val:.2f} cm")

    print()
    print_separator()
    print(f"  PREDICTION  [{model_name}]")
    print_separator()
    print(f"    --> Predicted Species : {predicted_name.upper()}")
    if confidence is not None:
        print(f"    --> Confidence        : {confidence:.2f}%")
    print_separator()


# -----------------------------------------------------------------------------
# Prediction Modes
# -----------------------------------------------------------------------------

def predict_manual(model, scaler, feature_names: list,
                   target_names: list, model_name: str = "Model") -> tuple:
    """
    Interactively prompt the user to enter one sample's feature values
    and display the prediction.

    Args:
        model        (estimator): Trained classifier.
        scaler       (StandardScaler): Fitted scaler.
        feature_names (list):    Feature column names.
        target_names  (list):    Class name strings.
        model_name    (str):     Name of the model used.

    Returns:
        tuple: (features_array, predicted_name, confidence)
    """
    print_banner("Manual Prediction — Enter Feature Values")
    print_info("Enter each measurement in centimetres (cm).")
    print_info("Typical Iris ranges: Sepal 4–8 cm  |  Petal 0.1–7 cm")
    print()

    values = []
    prompts = [
        ("Sepal Length", 0.0, 15.0),
        ("Sepal Width",  0.0, 15.0),
        ("Petal Length", 0.0, 15.0),
        ("Petal Width",  0.0, 15.0),
    ]

    for label, lo, hi in prompts:
        val = get_float_input(f"    Enter {label} (cm): ", min_val=lo, max_val=hi)
        values.append(val)

    features = np.array(values, dtype=float)

    start = time.perf_counter()
    name, conf, _ = predict_sample(model, scaler, features, target_names)
    elapsed = time.perf_counter() - start

    _display_prediction_result(features, feature_names, name, conf, target_names, model_name)
    print_info(f"Prediction time: {format_duration(elapsed)}")

    return features, name, conf


def predict_random(model, scaler, X_test: np.ndarray,
                   y_test: np.ndarray, feature_names: list,
                   target_names: list, model_name: str = "Model") -> None:
    """
    Randomly select one sample from the test set, predict it, and compare
    with the true label.

    Args:
        model        (estimator):  Trained classifier.
        scaler       (StandardScaler): Fitted scaler.
        X_test       (ndarray):    Unscaled test features.
        y_test       (ndarray):    True test labels.
        feature_names (list):      Feature column names.
        target_names  (list):      Class name strings.
        model_name    (str):       Model label.
    """
    print_banner("Random Test Sample Prediction")

    idx = np.random.randint(0, len(X_test))
    features   = X_test[idx]
    true_label = target_names[y_test[idx]]

    name, conf, _ = predict_sample(model, scaler, features, target_names)

    _display_prediction_result(features, feature_names, name, conf, target_names, model_name)

    print()
    print(f"    [TRUE]  True Label      : {true_label.upper()}")
    correct = "[CORRECT!]" if name == true_label else "[INCORRECT]"
    print(f"    {correct}")
    print_separator()


def predict_batch(model, scaler, feature_names: list,
                  target_names: list, model_name: str = "Model") -> list:
    """
    Collect multiple samples from the user and predict each one.

    The user enters each sample as a comma-separated list of 4 float values.
    Type 'done' to finish entering samples.

    Args:
        model         (estimator): Trained classifier.
        scaler        (StandardScaler): Fitted scaler.
        feature_names (list):      Feature column names.
        target_names  (list):      Class name strings.
        model_name    (str):       Model label.

    Returns:
        list: List of dicts with keys 'features', 'prediction', 'confidence'.
    """
    print_banner("Batch Prediction — Multiple Samples")
    print_info("Enter each sample as 4 comma-separated values:")
    print_info("  <sepal_length>, <sepal_width>, <petal_length>, <petal_width>")
    print_info("Type 'done' when finished, 'skip' to skip a line.\n")

    records = []
    sample_num = 1

    while True:
        raw = input(f"  Sample {sample_num}: ").strip()
        if raw.lower() in ("done", "exit", "quit", ""):
            break
        if raw.lower() == "skip":
            sample_num += 1
            continue

        try:
            parts = [float(v.strip()) for v in raw.split(",")]
            if len(parts) != 4:
                print_warning("Please enter exactly 4 values separated by commas.")
                continue

            features = np.array(parts, dtype=float)
            name, conf, _ = predict_sample(model, scaler, features, target_names)

            record = {
                "sample":    sample_num,
                "sepal_length": parts[0],
                "sepal_width":  parts[1],
                "petal_length": parts[2],
                "petal_width":  parts[3],
                "prediction":   name,
                "confidence":   f"{conf:.2f}%" if conf is not None else "N/A",
            }
            records.append(record)

            conf_str = f" ({conf:.1f}% confidence)" if conf is not None else ""
            print_success(f"  -> Prediction: {name.upper()}{conf_str}")
            sample_num += 1

        except ValueError:
            print_warning("Invalid input. Use numeric values separated by commas.")

    if records:
        print()
        print_section(f"Batch Summary — {len(records)} sample(s) predicted")
        _print_batch_table(records)

    return records


def _print_batch_table(records: list) -> None:
    """Print a tabular summary of batch prediction results."""
    col_w = [8, 13, 12, 13, 12, 15, 12]
    header = (f"  {'#':>{col_w[0]}} "
              f"{'Sep.Len':>{col_w[1]}} "
              f"{'Sep.Wid':>{col_w[2]}} "
              f"{'Pet.Len':>{col_w[3]}} "
              f"{'Pet.Wid':>{col_w[4]}} "
              f"{'Prediction':>{col_w[5]}} "
              f"{'Confidence':>{col_w[6]}}")
    print(header)
    print("  " + "-" * (sum(col_w) + len(col_w)))
    for r in records:
        print(
            f"  {r['sample']:>{col_w[0]}} "
            f"  {r['sepal_length']:>{col_w[1] - 2}.2f} "
            f"  {r['sepal_width']:>{col_w[2] - 2}.2f} "
            f"  {r['petal_length']:>{col_w[3] - 2}.2f} "
            f"  {r['petal_width']:>{col_w[4] - 2}.2f} "
            f"  {r['prediction']:>{col_w[5] - 2}} "
            f"  {r['confidence']:>{col_w[6] - 2}}"
        )


# -----------------------------------------------------------------------------
# Export Predictions to CSV
# -----------------------------------------------------------------------------

def export_predictions_to_csv(records: list,
                               filename: str = "predictions.csv") -> str:
    """
    Write a list of prediction records to a CSV file in the outputs/ directory.

    Args:
        records  (list): List of dicts as returned by predict_batch().
        filename (str):  Output file name. Default 'predictions.csv'.

    Returns:
        str: Absolute path of the written CSV file.
    """
    if not records:
        print_warning("No records to export.")
        return ""

    outputs_dir = get_outputs_dir()
    filepath = os.path.join(outputs_dir, filename)

    fieldnames = ["sample", "sepal_length", "sepal_width",
                  "petal_length", "petal_width", "prediction", "confidence"]

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print_success(f"Predictions exported -> {filepath}")
        return filepath

    except Exception as e:
        print_error(f"Failed to export CSV: {e}")
        raise


def export_test_predictions_to_csv(model, scaler,
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   feature_names: list, target_names: list,
                                   filename: str = "test_set_predictions.csv") -> str:
    """
    Predict all samples in the test set and save results to a CSV file.

    Args:
        model        (estimator): Trained classifier.
        scaler       (StandardScaler): Fitted scaler.
        X_test       (ndarray):  Unscaled test feature matrix.
        y_test       (ndarray):  True test labels.
        feature_names (list):    Feature column names.
        target_names  (list):    Class name strings.
        filename      (str):     Output file name.

    Returns:
        str: Path to the written CSV file.
    """
    print_section("Exporting Test Set Predictions to CSV")

    records = []
    for i in range(len(X_test)):
        features = X_test[i]
        name, conf, _ = predict_sample(model, scaler, features, target_names)
        true_name = target_names[y_test[i]]

        record = {
            "sample":       i + 1,
            "sepal_length": round(features[0], 2),
            "sepal_width":  round(features[1], 2),
            "petal_length": round(features[2], 2),
            "petal_width":  round(features[3], 2),
            "true_label":   true_name,
            "prediction":   name,
            "confidence":   f"{conf:.2f}%" if conf is not None else "N/A",
            "correct":      "Yes" if name == true_name else "No",
        }
        records.append(record)

    outputs_dir = get_outputs_dir()
    filepath = os.path.join(outputs_dir, filename)

    fieldnames = ["sample", "sepal_length", "sepal_width",
                  "petal_length", "petal_width",
                  "true_label", "prediction", "confidence", "correct"]

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        correct_count = sum(1 for r in records if r["correct"] == "Yes")
        print_success(f"Test set predictions exported -> {filepath}")
        print_info(f"{correct_count}/{len(records)} samples correctly predicted.")
        return filepath

    except Exception as e:
        print_error(f"Failed to export CSV: {e}")
        raise
