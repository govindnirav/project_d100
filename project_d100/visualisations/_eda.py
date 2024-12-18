from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dython.nominal import associations
from fitter import Fitter


def plot_corr(df: pl.DataFrame, index_var: str) -> plt.Figure:
    """Plots a correlation matrix heatmap.

    Args:
        df (pl.DataFrame): polars DataFrame

    Returns:
    - plt.Figure: the figure object
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

    return plt.gcf()


def plot_cramerv(
    df: pl.DataFrame,
    title: Optional[str] = "Cramér's V Correlation Matrix",
    index_var: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plots a Cramér's V correlation matrix using seaborn heatmap.

    Args:
        df (pl.DataFrame): The polars DataFrame.
        title (str, optional): Title of the plot.
        figsize (tuple, optional): Size of the plot (default is (10, 8)).
        filename (str, optional): File name to save the plot as PNG. If None, it will not be saved.

    Returns:
        plt.Figure: The figure object.
    """
    plt.figure(figsize=figsize)
    df = df.drop(index_var) if index_var else df

    # Calculate the Cramér's V matrix
    associations_matrix = associations(
        df.to_pandas(), nom_nom_assoc="cramer", compute_only=True
    )["corr"]

    # Plot the heatmap
    sns.heatmap(
        associations_matrix, annot=True, fmt=".2f", cmap="crest", cbar=True, square=True
    )
    plt.xticks(rotation=45)

    plt.title(title, fontsize=12)
    plt.gca().spines["right"].set_visible(False)  # Remove the right border
    plt.gca().spines["top"].set_visible(False)  # Remove the top border

    return plt.gcf()


def plot_count(
    df: pl.DataFrame,
    target_var: str,
    hue: Optional[str] = None,
    xticks: Optional[tuple] = None,
    fontsize: Optional[int] = 12,
) -> plt.Figure:
    """Plots a count of the target variable.

    Args:
        df (pl.DataFrame): polars DataFrame
        target_var (str): target variable
        hue (str): hue variable
        xticks (tuple): xticks range

    Returns:
    - plt.Figure: the figure object
    """
    plt.figure(figsize=(10, 8))
    sns.countplot(data=df, x=target_var, hue=hue, color="lightblue")
    if xticks is not None:
        tick_values = np.linspace(xticks[0], xticks[1], 11)
        plt.xticks(ticks=tick_values, labels=[str(int(tick)) for tick in tick_values])
    plt.title(f"Count of {target_var}", fontsize=fontsize)
    plt.xlabel(f"{target_var}", fontsize=fontsize)
    plt.ylabel("Count", fontsize=fontsize)

    plt.gca().spines["right"].set_visible(False)  # Remove the right border
    plt.gca().spines["top"].set_visible(False)  # Remove the top border

    return plt.gcf()


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


def plot_pairs(
    df: pl.DataFrame, vars: Optional[list[str]] = None, hue: Optional[str] = None
) -> plt.Figure:
    """Plots a pairplot for all numeric variables
        or specified variables in the DataFrame.

    Args:
        df (pl.DataFrame): polars DataFrame
        vars (list[str]): list of column names to plot
        hue (Optional[str]): column by which to color

    Returns:
    - plt.Figure: the figure object
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

    plt.gca().spines["right"].set_visible(False)  # Remove the right border
    plt.gca().spines["top"].set_visible(False)  # Remove the top border

    return plt.gcf()


def plot_kde(
    df: pl.DataFrame,
    col_name: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plots a kernel density estimate plot for a given column.

    Args:
        df (pl.DataFrame): polars DataFrame
        col_name (str): name of column to plot
        title (str): title of the plot

    Returns:
    - plt.Figure: the figure object
    """
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df.to_pandas(), x=col_name, hue=hue, fill=True, palette="crest")
    plt.title(title if title else f"Kernel Density Estimate of {col_name}")

    plt.gca().spines["right"].set_visible(False)  # Remove the right border
    plt.gca().spines["top"].set_visible(False)  # Remove the top border

    return plt.gcf()


def plot_violin(
    df: pl.DataFrame,
    y_var: Optional[str] = None,
    x_var: Optional[str] = None,
    hue: Optional[str] = None,
    orient: Optional[str] = "h",
    split: Optional[bool] = False,
    inner: Optional[str] = "quartile",
    gap: Optional[float] = 0,
) -> plt.Figure:
    """Plots a violin plot of the variable against x or y.

    Args:
        df (pl.DataFrame): Polars DataFrame
        y_var (str): Variable on the y-axis.
        x_var (str): Variable on the x-axis.
        hue (str): Column to color by.
        split (bool): Whether to split the violin by the hue variable.

    Returns:
    - plt.Figure: the figure object
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

    plt.gca().spines["right"].set_visible(False)  # Remove the right border
    plt.gca().spines["top"].set_visible(False)  # Remove the top border

    return plt.gcf()


def plot_scatter(
    df: pl.DataFrame, x_var: str, y_var: str, hue: Optional[str] = None
) -> plt.Figure:
    """Plots a scatter plot of the variables x_var and y_var.

    Args:
        df (pl.DataFrame): Polars DataFrame
        x_var (str): Variable on the x-axis.
        y_var (str): Variable on the y-axis.
        hue (str): Column to color by.

    Returns:
    - plt.Figure: the figure object
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df.to_pandas(), x=x_var, y=y_var, hue=hue, palette="crest")
    plt.title(f"Scatter Plot of {x_var} and {y_var}")

    plt.gca().spines["right"].set_visible(False)  # Remove the right border
    plt.gca().spines["top"].set_visible(False)  # Remove the top border

    return plt.gcf()


def plot_histogram(
    df: pl.DataFrame,
    col_name: str,
    bins: int = 30,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Frequency",
    ax: Optional[plt.Axes] = None,
    fontsize: Optional[int] = 12,
) -> None:
    """Plots a histogram using seaborn for visualization and polars for data handling.

    Args:
        df (pl.DataFrame): polars DataFrame
        col_name (str): name of column to plot
        bins (int): number of bins
        title (str): title of the histogram
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        ax (plt.Axes): Matplotlib axis object for subplots
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    sns.histplot(
        df[col_name], bins=bins, kde=True, ax=ax
    )  # KDE overlays the density plot
    ax.set_title(title if title else f"Histogram of {col_name}", fontsize=fontsize)
    ax.set_xlabel(xlabel if xlabel else col_name, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.spines["right"].set_visible(False)  # Remove the right border
    ax.spines["top"].set_visible(False)  # Remove the top border

    return ax


def plot_box(
    df: pl.DataFrame,
    y_var: Optional[str] = None,
    x_var: Optional[str] = None,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    whis: Optional[tuple | int | float] = 1.5,
    orient: str = "h",
    ax: Optional[plt.Axes] = None,
    fontsize: Optional[int] = 12,
) -> plt.Axes:
    """Plots changes in target variable across a time variable.

    Args:
        df (pl.DataFrame): polars DataFrame
        y_var (str): y variable
        x_var (str): x variable
        hue (str): column to color by
        whis (tuple | int | float): whisker length
        orient (str): orientation of the plot
        ax (plt.Axes): Matplotlib axis object for subplots
        fontsize (int): Font size for title and labels

    Returns:
        plt.Axes: The axis with the plot drawn.
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the boxplot on the specified axis
    sns.boxplot(
        data=df.to_pandas(),
        x=x_var,
        y=y_var,
        hue=hue,
        width=0.5,
        color="skyblue",
        whis=whis,
        orient=orient,
        ax=ax,
    )
    # Set labels and title with fontsize
    ax.set_xlabel(x_var if x_var else "", fontsize=fontsize)
    ax.set_ylabel(y_var if y_var else "", fontsize=fontsize)
    ax.set_title(
        title if title else f"Boxplot of {y_var} by {x_var}", fontsize=fontsize
    )

    # Remove unnecessary borders
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax
