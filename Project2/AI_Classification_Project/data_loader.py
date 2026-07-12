# =============================================================================
# data_loader.py - Dataset Loading and Exploratory Data Analysis (EDA)
# =============================================================================
# Description:
#   Handles all data ingestion and initial exploration tasks:
#   - Loads the Iris dataset from Scikit-Learn
#   - Converts it to a Pandas DataFrame
#   - Displays detailed EDA information (shape, types, statistics, missing values)
#   - Provides data for downstream modules
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from utils import print_banner, print_section, print_success, print_info, print_separator


# -----------------------------------------------------------------------------
# Dataset Loading
# -----------------------------------------------------------------------------

def load_dataset() -> tuple:
    """
    Load the Iris dataset from Scikit-Learn and return its components.

    The Iris dataset is a classic multiclass classification dataset with:
    - 150 samples  (50 per class)
    - 4 features   (sepal length, sepal width, petal length, petal width)
    - 3 classes    (Setosa, Versicolor, Virginica)

    Returns:
        tuple: (X, y, feature_names, target_names, dataframe)
            - X            (ndarray):  Feature matrix of shape (150, 4)
            - y            (ndarray):  Target label array of shape (150,)
            - feature_names (list):    List of 4 feature name strings
            - target_names  (list):    List of 3 class name strings
            - dataframe     (DataFrame): Combined features + target DataFrame
    """
    try:
        iris = load_iris()

        # -- Build a clean Pandas DataFrame including the target column ------
        dataframe = pd.DataFrame(
            data=iris.data,
            columns=iris.feature_names
        )
        dataframe["species"] = iris.target
        dataframe["species_name"] = dataframe["species"].apply(
            lambda idx: iris.target_names[idx]
        )

        return (
            iris.data,
            iris.target,
            list(iris.feature_names),
            list(iris.target_names),
            dataframe,
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load the Iris dataset: {e}")


# -----------------------------------------------------------------------------
# Exploratory Data Analysis (EDA) Display
# -----------------------------------------------------------------------------

def display_basic_info(X: np.ndarray, y: np.ndarray,
                       feature_names: list, target_names: list,
                       dataframe: pd.DataFrame) -> None:
    """
    Display basic dataset information to the console.

    Args:
        X            (ndarray):   Feature matrix.
        y            (ndarray):   Target array.
        feature_names (list):     Feature column names.
        target_names  (list):     Class label names.
        dataframe     (DataFrame): Combined dataset DataFrame.
    """
    print_banner("Dataset Basic Information")

    # -- Row & Column count ---------------------------------------------------
    print_section("Dataset Dimensions")
    print_info(f"Number of Rows    : {X.shape[0]}")
    print_info(f"Number of Columns : {X.shape[1]} features + 1 target")

    # -- Feature names --------------------------------------------------------
    print_section("Feature Names")
    for i, name in enumerate(feature_names, start=1):
        print_info(f"  Feature {i}: {name}")

    # -- Target class names ---------------------------------------------------
    print_section("Target Class Names")
    for i, name in enumerate(target_names, start=0):
        print_info(f"  Class {i}: {name}")

    # -- Class distribution ---------------------------------------------------
    print_section("Class Distribution")
    unique, counts = np.unique(y, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        print_info(f"  {target_names[cls_idx]:<15} : {count} samples")

    # -- First 5 rows ---------------------------------------------------------
    print_section("First 5 Rows of Dataset")
    print(dataframe.head().to_string(index=True))


def display_eda(dataframe: pd.DataFrame) -> None:
    """
    Display comprehensive exploratory data analysis for the dataset.

    Shows data types, missing values, and a full statistical summary.

    Args:
        dataframe (DataFrame): The combined Iris DataFrame.
    """
    print_banner("Exploratory Data Analysis (EDA)")

    # -- Shape ----------------------------------------------------------------
    print_section("Dataset Shape")
    print_info(f"Shape: {dataframe.shape[0]} rows × {dataframe.shape[1]} columns")

    # -- Data types -----------------------------------------------------------
    print_section("Data Types")
    print(dataframe.dtypes.to_string())

    # -- Missing values -------------------------------------------------------
    print_section("Missing Values")
    missing = dataframe.isnull().sum()
    if missing.sum() == 0:
        print_success("No missing values found! The dataset is clean.")
    else:
        print(missing[missing > 0].to_string())

    # -- Statistical summary --------------------------------------------------
    print_section("Statistical Summary (Numerical Features)")
    # Only display numeric feature columns (exclude species label columns)
    numeric_df = dataframe.drop(columns=["species", "species_name"])
    print(numeric_df.describe().round(3).to_string())

    # -- Per-class means ------------------------------------------------------
    print_section("Mean Feature Values per Species")
    grouped = dataframe.groupby("species_name")[
        [col for col in dataframe.columns if col not in ("species", "species_name")]
    ].mean().round(3)
    print(grouped.to_string())

    print_success("EDA complete.")


def get_feature_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Return the feature-only DataFrame (without the target columns).

    Args:
        dataframe (DataFrame): The full dataset DataFrame.

    Returns:
        DataFrame: DataFrame containing only the 4 feature columns.
    """
    return dataframe.drop(columns=["species", "species_name"])
