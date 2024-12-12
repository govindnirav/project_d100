from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from fitter import Fitter


def plot_corr(df: pl.DataFrame, index_var: str) -> None:
    """Plots a correlation matrix heatmap.

    Args:
        df (pd.DataFrame): pandas DataFrame
    """
    df = df.drop(index_var)
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="flare",
        square=True,
        cbar=True,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
    )
    plt.title("Feature Correlation Matrix")
    plt.show()


def plot_count(
    df: pl.DataFrame,
    target_var: str,
    hue: Optional[str] = None,
    xticks: Optional[tuple] = None,
) -> None:
    """Plots a count of the target variable.

    Args:
        df (pl.DataFrame): polars DataFrame
        target_var (str): target variable
        hue (str): hue variable
        xticks (tuple): xticks range
    """
    plt.figure(figsize=(10, 8))
    sns.countplot(data=df, x=target_var, hue=hue, color="lightblue")
    if xticks is not None:
        tick_values = np.linspace(xticks[0], xticks[1], 11)
        plt.xticks(ticks=tick_values, labels=[str(int(tick)) for tick in tick_values])
    plt.title(f"Count of {target_var}")
    plt.show()


def plot_dist(df: pl.DataFrame, target_var: str, dist: Optional[list] = None) -> None:
    """Plots a fit of the target variable against a list of distributions.

    Args:
        df (pl.DataFrame): polars DataFrame
        target_var (str): target variable
        dist (list): list of distributions to fit. if None, all distributions are used
    """
    f = Fitter(df[target_var].to_numpy(), distributions=dist)
    plt.figure(figsize=(10, 8))
    f.fit()
    f.summary()
    plt.show()


def plot_box(
    df: pl.DataFrame,
    y_var: Optional[str] = None,
    x_var: Optional[str] = None,
    hue: Optional[str] = None,
    whis: tuple | int | float = 1.5,
    orient: str = "h",
) -> None:
    """Plots a changes in target variable across a time variable.
    e.g number of rentals given the day of the week.

    Args:
        df (pl.DataFrame): polars DataFrame
        target_var (str): target variable
        time_var (str): time variable
    """
    plt.figure(figsize=(10, 8))
    sns.boxplot(
        data=df,
        x=x_var,
        y=y_var,
        color="lightgreen",
        hue=hue,
        width=0.5,
        whis=whis,
        orient=orient,
    )
    if orient == "h":
        plt.title(f"Changes in {x_var} over {y_var}")
    else:
        plt.title(f"Changes in {y_var} over {x_var}")
    plt.show()
