from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dython.nominal import associations
from fitter import Fitter


def plot_corr(df: pl.DataFrame, index_var: str) -> None:
    """Plots a correlation matrix heatmap.

    Args:
        df (pl.DataFrame): polars DataFrame
    """
    df = df.drop(index_var)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data=df.corr(),
        cmap="flare",
        square=True,
        cbar=True,
        xticklabels=df.corr().columns,
        yticklabels=df.corr().columns,
    )
    plt.title("Feature Correlation Matrix")
    plt.show()


def plot_cramerv(df: pl.DataFrame, index_var: str) -> None:
    """Plots a Cramer's V correlation matrix heatmap.
        Plots Cramer's V instead of Pearson's correlation for categorical variables.

    Args:
        df (pl.DataFrame): polars DataFrame
        index_var (str): index variable
    """
    df = df.drop(index_var)
    df = df.to_pandas()
    associations(df, nom_nom_assoc="cramer", figsize=(15, 15))


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


def plot_dist(
    df: pl.DataFrame, target_var: str, dist: Optional[list[str]] = None
) -> None:
    """Plots a fit of the target variable against a list of distributions.

    Args:
        df (pl.DataFrame): polars DataFrame
        target_var (str): target variable
        dist (list): list of distributions to fit. if None, all distributions are used
    """
    f = Fitter(df[target_var].to_numpy(), distributions=dist)
    plt.figure(figsize=(10, 8))
    f.fit()
    plt.title(f"Distribution of {target_var}, best fit: {f.get_best()}")
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


def plot_pairs(
    df: pl.DataFrame, vars: Optional[list[str]] = None, hue: Optional[str] = None
) -> None:
    """Plots a pairplot for all numeric variables
        or specified variables in the DataFrame.

    Args:
        df (pl.DataFrame): polars DataFrame
        vars (list[str]): list of column names to plot
        hue (Optional[str]): column by which to color
    """
    if vars is None:
        vars = [
            col
            for col, dtype in zip(df.columns, df.dtypes)
            if dtype in [pl.Float64, pl.Int64]
        ]

    sns.pairplot(
        df.to_pandas(),  # Does not support polars DataFrame
        vars=vars,
        hue=hue,
        corner=True,  # Removes duplicate plots
        palette="coolwarm",
    )
    plt.show()


def plot_histogram(
    df: pl.DataFrame,
    col_name: str,
    bins: int = 30,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Frequency",
) -> None:
    """Plots a histogram using seaborn for visualization and polars for data handling.

    Args:
    - df (pl.DataFrame): polars DataFrame
    - column (str): name of column to plot
    - bins (int): number of bins
    - title (str): title of the histogram
    - xlabel (str): x-axis label
    - ylabel (str): y-axis label
    """
    plt.figure(figsize=(10, 8))
    sns.histplot(
        df[col_name], bins=bins, kde=True
    )  # KDE option overlays the density plot
    plt.title(title if title else f"Histogram of {col_name}")
    plt.xlabel(xlabel if xlabel else col_name)
    plt.ylabel(ylabel)
    plt.show()


def plot_kde(
    df: pl.DataFrame,
    col_name: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Plots a kernel density estimate plot for a given column.

    Args:
    - df (pl.DataFrame): polars DataFrame
    - col_name (str): name of column to plot
    - title (str): title of the plot
    """
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df.to_pandas(), x=col_name, hue=hue, fill=True, palette="crest")
    plt.title(title if title else f"Kernel Density Estimate of {col_name}")
    plt.show()


def plot_violin(
    df: pl.DataFrame,
    y_var: Optional[str] = None,
    x_var: Optional[str] = None,
    hue: Optional[str] = None,
    orient: Optional[str] = "h",
    split: Optional[bool] = False,
    inner: Optional[str] = "quartile",
    gap: Optional[float] = 0,
) -> None:
    """Plots a violin plot of the variable against x or y.

    Args:
        df (pl.DataFrame): Polars DataFrame
        y_var (str): Variable on the y-axis.
        x_var (str): Variable on the x-axis.
        hue (str): Column to color by.
        split (bool): Whether to split the violin by the hue variable.
    """
    plt.figure(figsize=(10, 8))
    sns.violinplot(
        data=df.to_pandas(),
        x=x_var,
        y=y_var,
        hue=hue,
        split=split,
        orient=orient,
        gap=gap,
        saturation=1,
        inner=inner,
        linecolor="grey",
        palette="crest",
    )
    # Title
    if x_var is None:
        plt.title(f"Violin Plot for {y_var}")
    elif y_var is None:
        plt.title(f"Violin Plot for {x_var}")
    else:
        if orient == "h":
            plt.title(f"Violin Plot for {x_var} by {y_var}")
        else:
            plt.title(f"Violin Plot for {y_var} by {x_var}")
    plt.show()


def plot_scatter(
    df: pl.DataFrame, x_var: str, y_var: str, hue: Optional[str] = None
) -> None:
    """Plots a scatter plot of the variables x_var and y_var.

    Args:
        df (pl.DataFrame): Polars DataFrame
        x_var (str): Variable on the x-axis.
        y_var (str): Variable on the y-axis.
        hue (str): Column to color by.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df.to_pandas(), x=x_var, y=y_var, hue=hue, palette="crest")
    plt.title(f"Scatter Plot of {x_var} and {y_var}")
    plt.show()
