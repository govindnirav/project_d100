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

    print("\nData Types")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"{col}: {dtype}")

    print("\nMissing Values")
    print(df.null_count())  # All missing values are stored as null in a pl.DataFrame

    print("\nUnique Values")
    print(df["instant"].n_unique())

    print("\nStatistical Summary")
    print(df.describe())


def check_sum(df: pl.DataFrame, sum_var: str, var1: str, var2: str) -> None:
    """Checks if the sum of two variables is equal to the sum variable.

    Args:
        df (pl.DataFrame): polars DataFrame
        sum_var (str): variable that is the sum of var1 and var2
        var1 (str): first variable column
        var2 (str): second variable column

    Raises:
        AssertionError: for specific rows if the
        sum of var1 and var2 is not equal to sum_var
    """
    check = df[var1] + df[var2] == df[sum_var]

    failing_rows = df.filter(~check)

    assert check.all(), f"The following rows fail the condition:\n{failing_rows}"
