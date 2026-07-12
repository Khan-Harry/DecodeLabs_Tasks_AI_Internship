# =============================================================================
# preprocessing.py - Data Preprocessing
# =============================================================================
# Description:
#   Handles all data preparation tasks required before model training:
#   - Train/Test split (80/20) with a fixed random seed for reproducibility
#   - Feature scaling using StandardScaler
#   - Prints educational explanations on why preprocessing matters
# =============================================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import print_banner, print_section, print_success, print_info


# -----------------------------------------------------------------------------
# Train/Test Split
# -----------------------------------------------------------------------------

def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Split feature matrix and target array into train and test subsets.

    Uses an 80/20 stratified split so each class is proportionally
    represented in both subsets. The random_state ensures reproducibility—
    running the script twice will always produce the same split.

    Args:
        X            (ndarray): Full feature matrix of shape (n_samples, n_features).
        y            (ndarray): Full target array of shape (n_samples,).
        test_size    (float):   Fraction of data reserved for testing. Default 0.20.
        random_state (int):     Seed for the random number generator.  Default 42.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            - X_train (ndarray): Training features  — 80% of data
            - X_test  (ndarray): Testing features   — 20% of data
            - y_train (ndarray): Training labels    — 80% of data
            - y_test  (ndarray): Testing labels     — 20% of data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # Maintain class balance in both splits
    )
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# Feature Scaling
# -----------------------------------------------------------------------------

def scale_features(X_train: np.ndarray,
                   X_test: np.ndarray) -> tuple:
    """
    Standardise features so each has zero mean and unit variance.

    WHY SCALING?
    -------------------------------------------------------------------------
    Some algorithms (Logistic Regression, KNN) are sensitive to the magnitude
    of feature values. If one feature ranges 0–8 and another 0–1, the first
    will dominate distance and gradient calculations. StandardScaler removes
    this bias by converting every feature to the same z-score scale:

        z = (x − mean) / std_deviation

    IMPORTANT: The scaler is *fit only on training data* and then used to
    transform both training and test data. Fitting on the full dataset would
    cause "data leakage"—the model would implicitly see test statistics
    during training, leading to falsely optimistic evaluation results.

    Args:
        X_train (ndarray): Training feature matrix (used to fit the scaler).
        X_test  (ndarray): Testing feature matrix  (only transformed, not fit).

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
            - X_train_scaled (ndarray):       Scaled training features.
            - X_test_scaled  (ndarray):       Scaled testing features.
            - scaler         (StandardScaler): Fitted scaler (reuse for new samples).
    """
    scaler = StandardScaler()

    # Fit ONLY on training data, then transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# -----------------------------------------------------------------------------
# Educational Explanation
# -----------------------------------------------------------------------------

def explain_preprocessing() -> None:
    """
    Print a beginner-friendly explanation of why preprocessing is necessary.
    """
    print_banner("Why Preprocessing Matters")

    print_section("1. Train/Test Split")
    print_info("We divide the dataset into two non-overlapping subsets:")
    print_info("  * Training set (80%) — used to teach the model patterns.")
    print_info("  * Testing  set (20%) — held back to evaluate generalisation.")
    print_info("Without a test set we cannot know if the model just memorised")
    print_info("the training data (overfitting) or truly learned useful patterns.")

    print_section("2. Stratified Splitting")
    print_info("Setting stratify=y ensures each class (Setosa, Versicolor,")
    print_info("Virginica) is represented proportionally in both subsets.")
    print_info("This prevents a lucky/unlucky split from skewing accuracy scores.")

    print_section("3. Feature Scaling (StandardScaler)")
    print_info("Standardisation transforms each feature x to a z-score:")
    print_info("       z = (x - mean) / standard_deviation")
    print_info("Benefits:")
    print_info("  * KNN: computes Euclidean distances — unscaled features bias it.")
    print_info("  * Logistic Regression: gradient descent converges faster.")
    print_info("  * Decision Tree: tree splits are scale-invariant, but scaling")
    print_info("    is applied for a fair comparison across all models.")

    print_section("4. Preventing Data Leakage")
    print_info("The scaler is fitted ONLY on training data, then applied to both")
    print_info("train and test sets. Fitting on all data would expose test")
    print_info("statistics to the model, producing dishonestly high accuracy.")


# -----------------------------------------------------------------------------
# Full Preprocessing Pipeline
# -----------------------------------------------------------------------------

def preprocess(X: np.ndarray, y: np.ndarray,
               verbose: bool = True) -> tuple:
    """
    Run the full preprocessing pipeline: split, then scale.

    Args:
        X       (ndarray): Raw feature matrix.
        y       (ndarray): Target array.
        verbose (bool):    If True, print split statistics and explain steps.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler)
    """
    if verbose:
        explain_preprocessing()

    # Step 1: Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 2: Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    if verbose:
        print_section("Split Summary")
        print_success(f"Total samples  : {len(X)}")
        print_success(f"Training set   : {len(X_train)} samples (80%)")
        print_success(f"Testing  set   : {len(X_test)}  samples (20%)")
        print_success("Feature scaling applied with StandardScaler.")
        print_success("Preprocessing complete — data is ready for model training.")

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler
