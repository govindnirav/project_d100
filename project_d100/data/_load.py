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

    print("\nMissing Values")
    print(df.null_count())  # All missing values are stored as null in a pl.DataFrame

    print("\nUnique Values")
    print(df["instant"].n_unique())

    print("\nStatistical Summary")
    print(df.describe())
