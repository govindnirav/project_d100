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
) -> plt.Figure:
    """Plot actual vs predicted outcomes for two models side by side

    Args:
        y_test1 (pd.Series): actual outcomes for model 1
        y_preds1 (pd.Series): predicted outcomes for model 1
        model_name1 (str): name of the first model
        y_test2 (pd.Series): actual outcomes for model 2
        y_preds2 (pd.Series): predicted outcomes for model 2
        model_name2 (str): name of the second model
        axis (tuple): axis limits

    Returns:
        plt.Figure: Matplotlib figure
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
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["top"].set_visible(False)
    axes[0].set_xlim(axis[0], axis[1])
    axes[0].set_ylim(axis[0], axis[1])
    axes[0].set_title(f"Actual vs Predicted: {model_name1}", fontsize=20)
    axes[0].set_xlabel("Predicted", fontsize=20)
    axes[0].set_ylabel("Actual", fontsize=20)

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
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[1].set_xlim(axis[0], axis[1])
    axes[1].set_ylim(axis[0], axis[1])
    axes[1].set_title(f"Actual vs Predicted: {model_name2}", fontsize=20)
    axes[1].set_xlabel("Predicted", fontsize=20)
    axes[1].set_ylabel("Actual", fontsize=20)

    plt.tight_layout()
    return plt.gcf()


def plot_lorenz_curve(
    y_test1: np.ndarray | pd.Series,
    y_preds1: np.ndarray | pd.Series,
    model_name1: str,
    y_test2: np.ndarray | pd.Series,
    y_preds2: np.ndarray | pd.Series,
    model_name2: str,
) -> plt.Figure:
    """Plot Lorenz curves for two models superposed

    Args:
        y_test1 (np.ndarray | pd.Series): actual outcomes for model 1
        y_preds1 (np.ndarray | pd.Series): predicted outcomes for model 1
        model_name1 (str): name of the first model
        y_test2 (np.ndarray | pd.Series): actual outcomes for model 2
        y_preds2 (np.ndarray | pd.Series): predicted outcomes for model 2
        model_name2 (str): name of the second model

    Returns:
        plt.Figure: Matplotlib figure
    """
    ordered_samples1, cum_actuals1 = _calculate_lorenz(y_test1, y_preds1)
    ordered_samples2, cum_actuals2 = _calculate_lorenz(y_test2, y_preds2)
    # Oracle model
    ordered_samples3, cum_actuals3 = _calculate_lorenz(y_test1, y_test1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ordered_samples1, cum_actuals1, label=f"{model_name1}", color="green")
    ax.plot(ordered_samples2, cum_actuals2, label=f"{model_name2}", color="blue")
    ax.plot(ordered_samples3, cum_actuals3, label="Oracle", color="black")
    ax.plot([0, 1], [0, 1], linestyle="-", color="red")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlabel("Fraction of \n bike rentals from lowest to highest")
    ax.set_ylabel("Fraction of total bike rentals")
    ax.set_title("Superposed Lorenz Curves (with Oracle)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()

    return plt.gcf()


def plot_variable_importance(
    explainer: dx.Explainer, model_name: str, max_vars: int = 5
) -> plt.Figure:
    """Plot feature relevance for a model

    Args:
        explainer (dx.Explainer): Dalex explainer object
        model_name (str): model name
    """
    vi = explainer.model_parts()
    fig = vi.plot(
        title=f"Feature Importance: {model_name}", max_vars=max_vars, show=False
    )

    return fig


def plot_partial_dependence(
    lgbmexplainer: dx.Explainer,
    glmexplainer: dx.Explainer,
    features: list[str],
    type: Optional[str] = "partial",
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot partial dependence plots for a model

    Args:
        lgbmexplainer (dx.Explainer): LGBM Dalex explainer object
        glmexplainer (dx.Explainer): GLM Dalex explainer object
        features (list): list of features to plot
        type (str, optional): type of plot. Defaults to partial
                              can be "pdp" or "ale"
        title (str, optional): plot title

    """
    lgbm = lgbmexplainer.model_profile(type=type, label="LGBM")
    glm = glmexplainer.model_profile(type=type, label="GLM")

    if type == "pdp":
        title = "Partial Dependence Plots"
    elif type == "ale":
        title = "Accumulated Local Effect Plots"

    fig = lgbm.plot(glm, variables=features, title=title, show=False)

    return fig


def plot_shapley(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    model_name: str,
    max_display: Optional[int] = None,
    waterfall: Optional[bool] = False,
) -> plt.Figure:
    """
    Plot SHAP summary for the model, including beeswarm, waterfall, and scatter plots.

    Args:
        pipeline (Pipeline): Fitted pipeline model.
        X_test (pd.DataFrame): Test data for SHAP analysis.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    explainer = shap.Explainer(model, X_processed_df)
    shap_values = explainer(X_processed_df)

    max_display = max_display if max_display else len(feature_names)
    print("\nBeeswarm Plot")
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Summary: {model_name}")

    return plt.gcf()
