import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
