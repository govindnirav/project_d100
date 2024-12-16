import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_preds_actual(
    y_test1: np.ndarray | pd.Series,
    y_preds1: np.ndarray | pd.Series,
    model_name1: str,
    y_test2: np.ndarray | pd.Series,
    y_preds2: np.ndarray | pd.Series,
    model_name2: str,
    axis: tuple[float, float],
) -> None:
    """Plot actual vs predicted outcomes for two models side by side.

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
