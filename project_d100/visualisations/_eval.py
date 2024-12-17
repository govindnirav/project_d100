from typing import Optional

import dalex as dx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline

from project_d100.evaluation import _calculate_lorenz


def plot_preds_actual(
    y_test1: np.ndarray | pd.Series,
    y_preds1: np.ndarray | pd.Series,
    model_name1: str,
    y_test2: np.ndarray | pd.Series,
    y_preds2: np.ndarray | pd.Series,
    model_name2: str,
    axis: tuple[float, float],
) -> None:
    """Plot actual vs predicted outcomes for two models side by side

    Args:
        y_test1 (pd.Series): actual outcomes for model 1
        y_preds1 (pd.Series): predicted outcomes for model 1
        model_name1 (str): name of the first model
        y_test2 (pd.Series): actual outcomes for model 2
        y_preds2 (pd.Series): predicted outcomes for model 2
        model_name2 (str): name of the second model
        axis (tuple): axis limits
    """
    axis_line = np.linspace(axis[0], axis[1], 10)

    diff1 = abs(y_test1 - y_preds1)
    diff2 = abs(y_test2 - y_preds2)

    # Set up a side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    # Model 1
    sns.scatterplot(
        x=y_preds1,
        y=y_test1,
        hue=diff1,
        palette="crest",
        alpha=0.5,
        edgecolor=None,
        legend=False,
        ax=axes[0],
    )
    sns.lineplot(x=axis_line, y=axis_line, color="red", ax=axes[0])
    axes[0].set_xlim(axis[0], axis[1])
    axes[0].set_ylim(axis[0], axis[1])
    axes[0].set_title(f"Actual vs Predicted: {model_name1}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Model 2
    sns.scatterplot(
        x=y_preds2,
        y=y_test2,
        hue=diff2,
        palette="crest",
        alpha=0.5,
        edgecolor=None,
        legend=False,
        ax=axes[1],
    )
    sns.lineplot(x=axis_line, y=axis_line, color="red", ax=axes[1])
    axes[1].set_xlim(axis[0], axis[1])
    axes[1].set_ylim(axis[0], axis[1])
    axes[1].set_title(f"Actual vs Predicted: {model_name2}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


def plot_lorenz_curve(
    y_test1: np.ndarray | pd.Series,
    y_preds1: np.ndarray | pd.Series,
    model_name1: str,
    y_test2: np.ndarray | pd.Series,
    y_preds2: np.ndarray | pd.Series,
    model_name2: str,
) -> None:
    """Plot Lorenz curves for two models superposed

    Args:
        y_test1 (np.ndarray | pd.Series): actual outcomes for model 1
        y_preds1 (np.ndarray | pd.Series): predicted outcomes for model 1
        model_name1 (str): name of the first model
        y_test2 (np.ndarray | pd.Series): actual outcomes for model 2
        y_preds2 (np.ndarray | pd.Series): predicted outcomes for model 2
         model_name2 (str): name of the second model
    """
    # Calculate Lorenz curve points for both models
    ordered_samples1, cum_actuals1 = _calculate_lorenz(y_test1, y_preds1)

    ordered_samples2, cum_actuals2 = _calculate_lorenz(y_test2, y_preds2)
    # Oracle model
    ordered_samples3, cum_actuals3 = _calculate_lorenz(y_test1, y_test1)

    # Plot the Lorenz curves
    plt.figure(figsize=(8, 8))
    plt.plot(ordered_samples1, cum_actuals1, label=f"{model_name1}", color="blue")
    plt.plot(ordered_samples2, cum_actuals2, label=f"{model_name2}", color="green")
    plt.plot(ordered_samples3, cum_actuals3, label="Oracle", color="black")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red")

    # Plot annotations
    plt.xlabel("Fraction of \n bike rentals from lowest to highest")
    plt.ylabel("Fraction of total bike rentals")
    plt.title("Superposed Lorenz Curves (with Oracle)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_variable_importance(explainer: dx.Explainer, model_name: str) -> None:
    """Plot feature relevance for a model

    Args:
        explainer (dx.Explainer): Dalex explainer object
        model_name (str): model name
    """
    vi = explainer.model_parts()
    vi.plot(title=f"Feature Importance: {model_name}")
    plt.show()


def plot_partial_dependence(
    lgbmexplainer: dx.Explainer,
    glmexplainer: dx.Explainer,
    features: list[str],
    type: Optional[str] = "partial",
) -> None:
    """Plot partial dependence plots for a model

    Args:
        lgbmexplainer (dx.Explainer): LGBM Dalex explainer object
        glmexplainer (dx.Explainer): GLM Dalex explainer object
        features (list): list of features to plot
        type (str, optional): type of plot. Defaults to partial
                              can be "pdp" or "ale"

    """
    lgbm = lgbmexplainer.model_profile(type=type, label="LGBM")
    glm = glmexplainer.model_profile(type=type, label="GLM")

    if type == "pdp":
        title = "Partial Dependence Plots"
    elif type == "ale":
        title = "Accumulated Local Effect Plots"

    lgbm.plot(glm, variables=features, title=title)


def plot_shapley(pipeline: Pipeline, X_test: pd.DataFrame) -> None:
    """
    Plot SHAP summary for the model, including beeswarm, waterfall, and scatter plots.

    Args:
        pipeline (Pipeline): Fitted pipeline model.
        X_test (pd.DataFrame): Test data for SHAP analysis.
    """
    # Sample the test set to improve computation speed
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    # Create SHAP explainer
    explainer = shap.Explainer(model, X_processed_df)
    shap_values = explainer(X_processed_df)

    # Beeswarm plot: Shows SHAP values for all features across all samples
    print("\nBeeswarm Plot")
    shap.plots.beeswarm(shap_values, max_display=len(feature_names))
    plt.show()

    # Waterfall plot: Shows how SHAP values push the prediction for a single observation
    sample_ind = 0  # First observation from the sample
    print("\nWaterfall Plot (for one observation)")
    shap.plots.waterfall(shap_values[sample_ind])
    plt.show()
