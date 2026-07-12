# =============================================================================
# main.py - Main Entry Point and CLI Menu System
# =============================================================================
# Description:
#   Interactive command-line interface (CLI) that guides the user through the
#   complete supervised machine learning pipeline:
#
#   Menu Options:
#   [1]  Load & Explore Data
#   [2]  Run Preprocessing
#   [3]  Train Models
#   [4]  Evaluate & Compare Models
#   [5]  Generate All Visualisations
#   [6]  Prediction System
#   [7]  Run Full Pipeline (steps 1–5 automatically)
#   [0]  Exit
#
#   State is maintained in a shared AppState object so results from earlier
#   steps are available to later steps within the same session.
# =============================================================================

import sys
import os

# Force UTF-8 output so special characters render correctly on all platforms,
# including Windows terminals that default to CP1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# -- Make sure the project directory is on the Python path --------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    print_banner, print_section, print_info,
    print_success, print_warning, print_error,
    print_separator, print_menu_option, get_int_input, get_yes_no_input
)
from data_loader    import load_dataset, display_basic_info, display_eda
from preprocessing  import preprocess
from model          import (
    train_all_models, save_all_models, load_all_models,
    save_scaler, load_scaler, MODEL_NAMES
)
from evaluation     import run_full_evaluation
from visualization  import generate_all_plots
from prediction     import (
    predict_manual, predict_random, predict_batch,
    export_predictions_to_csv, export_test_predictions_to_csv
)


# =============================================================================
# Application State Container
# =============================================================================

class AppState:
    """
    Holds all shared data produced during the pipeline session.
    Passed between menu handlers to avoid global variables.
    """

    def __init__(self):
        # -- Dataset ----------------------------------------------------------
        self.X             = None   # Raw feature matrix (150, 4)
        self.y             = None   # Target labels (150,)
        self.feature_names = None   # List of 4 feature name strings
        self.target_names  = None   # List of 3 class name strings
        self.dataframe     = None   # Full Pandas DataFrame

        # -- Preprocessing -----------------------------------------------------
        self.X_train    = None   # Unscaled training features
        self.X_test     = None   # Unscaled test features
        self.y_train    = None   # Training labels
        self.y_test     = None   # Test labels
        self.X_train_sc = None   # Scaled training features
        self.X_test_sc  = None   # Scaled test features
        self.scaler     = None   # Fitted StandardScaler

        # -- Models -----------------------------------------------------------
        self.trained_models = None  # {key: (model, training_time)}

        # -- Evaluation --------------------------------------------------------
        self.eval_results = None    # {key: {accuracy, confusion_matrix, y_pred}}
        self.accuracies   = None    # {key: accuracy_float}
        self.best_model   = None    # key of best model


# =============================================================================
# Pipeline Step Handlers
# =============================================================================

def step_load_data(state: AppState) -> None:
    """Step 1: Load the Iris dataset and display EDA information."""
    print_banner("Step 1 — Load & Explore Dataset")

    X, y, feature_names, target_names, df = load_dataset()
    state.X             = X
    state.y             = y
    state.feature_names = feature_names
    state.target_names  = target_names
    state.dataframe     = df

    display_basic_info(X, y, feature_names, target_names, df)
    display_eda(df)

    print_success("Dataset loaded and exploration complete.")


def step_preprocess(state: AppState) -> None:
    """Step 2: Split data and apply feature scaling."""
    if state.X is None:
        print_warning("Dataset not loaded yet. Running Step 1 first...")
        step_load_data(state)

    print_banner("Step 2 — Data Preprocessing")

    (state.X_train, state.X_test,
     state.y_train, state.y_test,
     state.X_train_sc, state.X_test_sc,
     state.scaler) = preprocess(state.X, state.y, verbose=True)

    # Save the scaler so it can be reloaded later
    save_scaler(state.scaler)
    print_success("Preprocessing complete.")


def step_train_models(state: AppState) -> None:
    """Step 3: Train Decision Tree, Logistic Regression, and KNN."""
    if state.X_train_sc is None:
        print_warning("Data not preprocessed yet. Running Step 2 first...")
        step_preprocess(state)

    print_banner("Step 3 — Train Classification Models")

    state.trained_models = train_all_models(
        state.X_train_sc, state.y_train, verbose=True
    )

    # Save all trained models to disk
    save_all_models(state.trained_models)
    print_success("All models trained and saved.")


def step_evaluate(state: AppState) -> None:
    """Step 4: Evaluate all models, show comparison, and feature importance."""
    if state.trained_models is None:
        print_warning("Models not trained yet. Running Step 3 first...")
        step_train_models(state)

    print_banner("Step 4 — Model Evaluation & Comparison")

    (state.eval_results,
     state.accuracies,
     state.best_model) = run_full_evaluation(
        state.trained_models,
        state.X_test_sc,
        state.y_test,
        state.feature_names,
        state.target_names
    )
    print_success("Evaluation complete.")


def step_visualise(state: AppState) -> None:
    """Step 5: Generate and save all visualisation plots."""
    if state.eval_results is None:
        print_warning("Models not evaluated yet. Running Step 4 first...")
        step_evaluate(state)

    print_banner("Step 5 — Generate Visualisations")

    generate_all_plots(
        dataframe      = state.dataframe,
        target_names   = state.target_names,
        feature_names  = state.feature_names,
        trained_models = state.trained_models,
        eval_results   = state.eval_results,
        accuracies     = state.accuracies,
    )
    print_success("All plots saved to the 'plots/' directory.")


# =============================================================================
# Prediction Sub-Menu
# =============================================================================

def prediction_menu(state: AppState) -> None:
    """Interactive sub-menu for the prediction system."""
    # -- Ensure models and scaler are available -----------------------------
    if state.trained_models is None or state.scaler is None:
        print_info("Attempting to load saved models and scaler from disk...")
        state.trained_models = {}
        loaded = load_all_models()
        from model import train_decision_tree, train_logistic_regression, train_knn
        for key, model in loaded.items():
            if model is not None:
                state.trained_models[key] = (model, 0.0)

        state.scaler = load_scaler()

        if not state.trained_models or state.scaler is None:
            print_warning("No saved models found. Running full pipeline first...")
            step_load_data(state)
            step_preprocess(state)
            step_train_models(state)

    if state.feature_names is None:
        step_load_data(state)
    if state.X_test is None:
        step_preprocess(state)

    # -- Choose model to use ------------------------------------------------
    print_section("Select Model for Prediction")
    model_keys = list(state.trained_models.keys())
    for i, k in enumerate(model_keys, start=1):
        print_menu_option(i, MODEL_NAMES[k])
    model_choice = get_int_input("  Enter model number: ", list(range(1, len(model_keys) + 1)))
    selected_key = model_keys[model_choice - 1]
    selected_model, _ = state.trained_models[selected_key]
    model_label = MODEL_NAMES[selected_key]

    while True:
        print_banner("Prediction System")
        print_menu_option(1, "Manual Input — Enter feature values")
        print_menu_option(2, "Random Test Sample Prediction")
        print_menu_option(3, "Batch Prediction — Multiple samples")
        print_menu_option(4, "Export All Test Set Predictions to CSV")
        print_menu_option(0, "Return to Main Menu")
        print()

        choice = get_int_input("  Select an option: ", [0, 1, 2, 3, 4])

        if choice == 0:
            break

        elif choice == 1:
            _, name, conf = predict_manual(
                selected_model, state.scaler,
                state.feature_names, state.target_names,
                model_name=model_label
            )

        elif choice == 2:
            predict_random(
                selected_model, state.scaler,
                state.X_test, state.y_test,
                state.feature_names, state.target_names,
                model_name=model_label
            )

        elif choice == 3:
            records = predict_batch(
                selected_model, state.scaler,
                state.feature_names, state.target_names,
                model_name=model_label
            )
            if records and get_yes_no_input("  Export these predictions to CSV?"):
                export_predictions_to_csv(records)

        elif choice == 4:
            export_test_predictions_to_csv(
                selected_model, state.scaler,
                state.X_test, state.y_test,
                state.feature_names, state.target_names
            )


# =============================================================================
# Full Automated Pipeline
# =============================================================================

def run_full_pipeline(state: AppState) -> None:
    """
    Execute Steps 1–5 automatically in sequence without user interruption.
    Ideal for a first run or CI/CD-style execution.
    """
    print_banner("Running Full Pipeline (Steps 1 – 5)")
    step_load_data(state)
    step_preprocess(state)
    step_train_models(state)
    step_evaluate(state)
    step_visualise(state)
    print_banner("Full Pipeline Complete")
    print_success("All steps finished. Models saved. Plots saved.")
    print_info("Use the Prediction System menu (option 6) to make predictions.")


# =============================================================================
# Main Menu Loop
# =============================================================================

MENU_OPTIONS = {
    1: "Load & Explore Data",
    2: "Run Preprocessing",
    3: "Train All Models",
    4: "Evaluate & Compare Models",
    5: "Generate All Visualisations",
    6: "Prediction System",
    7: "Run Full Pipeline (Steps 1–5 Automatically)",
    0: "Exit",
}


def print_main_menu() -> None:
    """Display the application main menu."""
    print_banner("Data Classification Using Artificial Intelligence")
    print_info("Iris Flower Species Classifier  |  Internship Portfolio Project")
    print()
    for num, desc in MENU_OPTIONS.items():
        print_menu_option(num, desc)
    print()


def main() -> None:
    """
    Main application entry point.
    Runs the interactive CLI menu loop until the user selects 'Exit'.
    """
    state = AppState()

    print()
    print("=" * 70)
    print("  WELCOME TO THE AI DATA CLASSIFICATION PROJECT")
    print("  Built with Python  |  Scikit-Learn  |  Pandas  |  Matplotlib")
    print("=" * 70)
    print()

    while True:
        try:
            print_main_menu()
            choice = get_int_input(
                "  Enter your choice: ",
                list(MENU_OPTIONS.keys())
            )

            if choice == 0:
                print_banner("Thank You!")
                print_info("Project by: AI Internship Student")
                print_info("Technologies: Python, Scikit-Learn, Pandas, Matplotlib, NumPy")
                print_success("Exiting... Goodbye! 👋")
                sys.exit(0)

            elif choice == 1:
                step_load_data(state)

            elif choice == 2:
                step_preprocess(state)

            elif choice == 3:
                step_train_models(state)

            elif choice == 4:
                step_evaluate(state)

            elif choice == 5:
                step_visualise(state)

            elif choice == 6:
                prediction_menu(state)

            elif choice == 7:
                run_full_pipeline(state)

            input("\n  Press ENTER to return to the main menu...")

        except KeyboardInterrupt:
            print()
            print_warning("Interrupted by user (Ctrl+C). Exiting...")
            sys.exit(0)

        except Exception as exc:
            print_error(f"An unexpected error occurred: {exc}")
            print_info("Please try again or restart the program.")


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
