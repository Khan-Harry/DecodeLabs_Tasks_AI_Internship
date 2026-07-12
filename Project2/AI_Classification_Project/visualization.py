# =============================================================================
# visualization.py - Plot Generation
# =============================================================================
# Description:
#   Generates and saves all project visualisations using Matplotlib:
#   - Histogram of feature distributions
#   - Scatter plot of key feature pairs
#   - Pair plot (scatter matrix)
#   - Correlation heatmap
#   - Confusion matrix plot
#   - Decision Tree diagram
#   - Accuracy comparison bar chart
#
#   All plots are saved as .png files in the 'plots/' directory.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (no GUI window needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

from utils import print_success, print_info, print_section, get_plots_dir


# -----------------------------------------------------------------------------
# Colour Palette
# -----------------------------------------------------------------------------

# Accessible, visually distinct colour for each Iris species
CLASS_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]   # blue, orange, green
BG_COLOR     = "#F8F9FA"
GRID_COLOR   = "#E0E0E0"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.grid":        True,
    "grid.color":       GRID_COLOR,
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
})


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _save_figure(filename: str) -> str:
    """Save the current figure to the plots directory and close it."""
    plots_dir = get_plots_dir()
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print_success(f"Plot saved -> {filepath}")
    return filepath


# -----------------------------------------------------------------------------
# EDA Visualisations
# -----------------------------------------------------------------------------

def plot_histograms(dataframe: pd.DataFrame, target_names: list) -> str:
    """
    Plot overlapping histograms for each feature, coloured by species.

    Args:
        dataframe    (DataFrame): Full Iris DataFrame with 'species_name' column.
        target_names (list):      List of class name strings.

    Returns:
        str: File path of the saved plot.
    """
    feature_cols = [c for c in dataframe.columns
                    if c not in ("species", "species_name")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Feature Distributions by Species", fontsize=16, fontweight="bold", y=1.01)

    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        for j, species in enumerate(target_names):
            subset = dataframe[dataframe["species_name"] == species][col]
            axes[i].hist(subset, bins=15, alpha=0.65,
                         color=CLASS_COLORS[j], label=species, edgecolor="white")
        axes[i].set_title(col, fontsize=12)
        axes[i].set_xlabel("Value (cm)")
        axes[i].set_ylabel("Frequency")
        axes[i].legend(fontsize=9)

    plt.tight_layout()
    return _save_figure("01_feature_histograms.png")


def plot_scatter(dataframe: pd.DataFrame, target_names: list) -> str:
    """
    Plot a scatter of Petal Length vs Petal Width coloured by species.

    Args:
        dataframe    (DataFrame): Full Iris DataFrame.
        target_names (list):      Class name strings.

    Returns:
        str: File path of the saved plot.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for j, species in enumerate(target_names):
        subset = dataframe[dataframe["species_name"] == species]
        ax.scatter(
            subset["petal length (cm)"],
            subset["petal width (cm)"],
            c=CLASS_COLORS[j],
            label=species,
            alpha=0.80,
            edgecolors="white",
            linewidths=0.5,
            s=80
        )

    ax.set_title("Petal Length vs Petal Width by Species",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Petal Length (cm)", fontsize=12)
    ax.set_ylabel("Petal Width  (cm)", fontsize=12)
    ax.legend(title="Species", fontsize=10)

    plt.tight_layout()
    return _save_figure("02_scatter_petal.png")


def plot_pair_plot(dataframe: pd.DataFrame, target_names: list) -> str:
    """
    Generate a manual scatter-matrix (pair plot) for all feature combinations.

    Args:
        dataframe    (DataFrame): Full Iris DataFrame.
        target_names (list):      Class name strings.

    Returns:
        str: File path of the saved plot.
    """
    feature_cols = [c for c in dataframe.columns
                    if c not in ("species", "species_name")]
    n = len(feature_cols)

    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    fig.suptitle("Pair Plot — All Feature Combinations",
                 fontsize=16, fontweight="bold", y=1.01)

    for row in range(n):
        for col in range(n):
            ax = axes[row][col]
            if row == col:
                # Diagonal: histogram
                for j, species in enumerate(target_names):
                    subset = dataframe[dataframe["species_name"] == species][feature_cols[row]]
                    ax.hist(subset, bins=12, alpha=0.6, color=CLASS_COLORS[j])
            else:
                # Off-diagonal: scatter
                for j, species in enumerate(target_names):
                    subset = dataframe[dataframe["species_name"] == species]
                    ax.scatter(subset[feature_cols[col]],
                               subset[feature_cols[row]],
                               c=CLASS_COLORS[j], alpha=0.5, s=20)

            if row == n - 1:
                ax.set_xlabel(feature_cols[col], fontsize=7)
            if col == 0:
                ax.set_ylabel(feature_cols[row], fontsize=7)
            ax.tick_params(labelsize=6)

    # Legend
    handles = [mpatches.Patch(color=CLASS_COLORS[j], label=target_names[j])
               for j in range(len(target_names))]
    fig.legend(handles=handles, loc="upper right",
               fontsize=10, title="Species", framealpha=0.9)

    plt.tight_layout()
    return _save_figure("03_pair_plot.png")


def plot_correlation_heatmap(dataframe: pd.DataFrame) -> str:
    """
    Plot a correlation heatmap for all numeric feature columns.

    Args:
        dataframe (DataFrame): Full Iris DataFrame.

    Returns:
        str: File path of the saved plot.
    """
    feature_cols = [c for c in dataframe.columns
                    if c not in ("species", "species_name")]
    corr = dataframe[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)

    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(feature_cols, fontsize=9)

    # Annotate cells with correlation values
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=10, color="black" if abs(corr.iloc[i, j]) < 0.7 else "white")

    ax.set_title("Feature Correlation Heatmap",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    return _save_figure("04_correlation_heatmap.png")


# -----------------------------------------------------------------------------
# Evaluation Visualisations
# -----------------------------------------------------------------------------

def plot_confusion_matrix(y_test_or_cm, y_pred: np.ndarray,
                          target_names: list,
                          model_name: str = "Model") -> str:
    """
    Plot a styled confusion matrix heatmap.

    Args:
        y_test_or_cm (ndarray): Either the true test label array (1-D) OR
                                a pre-computed confusion matrix (2-D).
        y_pred       (ndarray): Predicted labels (used only when y_test_or_cm
                                is a 1-D label array).
        target_names (list):    Class name strings.
        model_name   (str):     Label to include in the title.

    Returns:
        str: File path of the saved plot.
    """
    # Accept either raw labels or a pre-computed confusion matrix
    if y_test_or_cm.ndim == 2:
        cm = y_test_or_cm           # already a confusion matrix
    else:
        cm = confusion_matrix(y_test_or_cm, y_pred)

    n_classes = len(target_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(target_names, fontsize=11)
    ax.set_yticklabels(target_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontsize=14, fontweight="bold", pad=15)

    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    return _save_figure(f"05_confusion_matrix_{safe_name}.png")


def plot_decision_tree(dt_model, feature_names: list,
                       target_names: list) -> str:
    """
    Visualise the trained Decision Tree structure.

    Args:
        dt_model      (DecisionTreeClassifier): Fitted Decision Tree model.
        feature_names (list): Feature column names.
        target_names  (list): Class name strings.

    Returns:
        str: File path of the saved plot.
    """
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
        impurity=True,
        proportion=False,
        precision=3,
    )
    ax.set_title("Decision Tree Visualisation",
                 fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    return _save_figure("06_decision_tree.png")


def plot_accuracy_comparison(accuracies: dict, training_times: dict = None) -> str:
    """
    Plot a bar chart comparing model accuracies (and optionally training times).

    Args:
        accuracies     (dict): {model_key: accuracy_float}.
        training_times (dict): {model_key: time_float} (optional).

    Returns:
        str: File path of the saved plot.
    """
    from model import MODEL_NAMES

    labels   = [MODEL_NAMES[k] for k in accuracies]
    values   = [acc * 100 for acc in accuracies.values()]
    colors   = CLASS_COLORS[:len(labels)]
    best_acc = max(values)

    n_plots  = 2 if training_times else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(12 if n_plots == 2 else 8, 6))
    if n_plots == 1:
        axes = [axes]

    # -- Accuracy bar chart ---------------------------------------------------
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white",
                       linewidth=1.5, width=0.5)
    axes[0].set_ylim(85, 105)
    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    axes[0].set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f"{val:.2f}%",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
        if val == best_acc:
            bar.set_edgecolor("#FFD700")
            bar.set_linewidth(3)

    # -- Training time bar chart (optional) -----------------------------------
    if training_times and n_plots == 2:
        time_values = [t * 1000 for t in training_times.values()]  # ms
        time_bars   = axes[1].bar(labels, time_values, color=colors,
                                  edgecolor="white", linewidth=1.5, width=0.5)
        axes[1].set_ylabel("Training Time (ms)", fontsize=12)
        axes[1].set_title("Model Training Time Comparison",
                           fontsize=14, fontweight="bold")
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

        for bar, val in zip(time_bars, time_values):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.02,
                         f"{val:.2f} ms",
                         ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    return _save_figure("07_accuracy_comparison.png")


def plot_feature_importance(dt_model, feature_names: list) -> str:
    """
    Plot a horizontal bar chart of Decision Tree feature importances.

    Args:
        dt_model      (DecisionTreeClassifier): Fitted Decision Tree.
        feature_names (list): Feature column names.

    Returns:
        str: File path of the saved plot.
    """
    importances = dt_model.feature_importances_
    sorted_idx  = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sorted_names, sorted_vals,
                   color=CLASS_COLORS[0], edgecolor="white", height=0.5)

    for bar, val in zip(bars, sorted_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=11)

    ax.set_xlabel("Feature Importance (Gini Reduction)", fontsize=12)
    ax.set_title("Decision Tree — Feature Importances",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(sorted_vals) * 1.25)
    plt.tight_layout()
    return _save_figure("08_feature_importance.png")


# -----------------------------------------------------------------------------
# Full Visualisation Pipeline
# -----------------------------------------------------------------------------

def generate_all_plots(dataframe: pd.DataFrame,
                       target_names: list,
                       feature_names: list,
                       trained_models: dict,
                       eval_results: dict,
                       accuracies: dict) -> list:
    """
    Run the complete visualisation pipeline and save all plots.

    Args:
        dataframe      (DataFrame): Full Iris DataFrame.
        target_names   (list):      Class name strings.
        feature_names  (list):      Feature column names.
        trained_models (dict):      Dict from train_all_models().
        eval_results   (dict):      Dict from run_full_evaluation().
        accuracies     (dict):      {model_key: accuracy_float}.

    Returns:
        list: Paths to all generated plot files.
    """
    print_section("Generating All Visualisations")
    paths = []

    # EDA plots
    paths.append(plot_histograms(dataframe, target_names))
    paths.append(plot_scatter(dataframe, target_names))
    paths.append(plot_pair_plot(dataframe, target_names))
    paths.append(plot_correlation_heatmap(dataframe))

    # Confusion matrix for each model (use pre-computed cm from eval_results)
    for key, (model, _) in trained_models.items():
        from model import MODEL_NAMES
        y_pred = eval_results[key]["y_pred"]
        paths.append(
            plot_confusion_matrix(
                eval_results[key]["confusion_matrix"],   # 2-D array -> used directly
                y_pred,
                target_names,
                model_name=MODEL_NAMES[key]
            )
        )

    # Decision Tree visualisation
    dt_model = trained_models["decision_tree"][0]
    paths.append(plot_decision_tree(dt_model, feature_names, target_names))

    # Accuracy & time comparison
    training_times = {k: t for k, (_, t) in trained_models.items()}
    paths.append(plot_accuracy_comparison(accuracies, training_times))

    # Feature importance
    paths.append(plot_feature_importance(dt_model, feature_names))

    print_success(f"All {len(paths)} plots generated and saved to 'plots/' directory.")
    return paths
