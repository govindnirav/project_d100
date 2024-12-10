from pathlib import Path

import polars as pl

RAW_DATA = Path(__file__).parent.parent.parent / "data" / "raw" / "hour.csv"
CLEAN_DATA = Path(__file__).parent.parent.parent / "data" / "clean"


def load_data() -> pl.DataFrame:
    """Loads the raw dataset from path above.

    Returns:
        pl.DataFrame: Raw dataset
    """
    df = pl.read_csv(RAW_DATA)
    return df


def summary(df: pl.DataFrame) -> None:
    """Generates summary statistics of the data

    Args:
        df (pl.DataFrame): polars DataFrame
    """
    print("\nData Shape (rows, columns)")
    print(df.shape)

    print("\nData Types of Columns")
    print(df.dtypes)

    print("\nStatistical Summary")
    print(df.describe())

    return df


def extract_day(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """Extracts day from colname, as an integer.
    Deletes colname.

    Args:
        df (pl.DataFrame): polars DataFrame

    Returns:
        pl.DataFrame: polars DataFrame
    """
    df = df.with_columns(
        pl.col(col_name).str.strptime(pl.Date, "%Y-%m-%d").dt.day().alias("day")
    )
    df = df.drop(col_name)
    return df


def denormalize(df: pl.DataFrame, col_name: str, value: float) -> pl.DataFrame:
    """Denormalizes a column by multiplying it by value.

    Args:
        df (pl.DataFrame): polars DataFrame

    Returns:
        pl.DataFrame: polars DataFrame
    """
    df = df.with_columns((pl.col(col_name) * value).alias(col_name))
    return df
