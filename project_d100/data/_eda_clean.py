import polars as pl


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


def ren_year(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """Rename values in colname from 0 and 1 to 2011 and 2012

    Args:
        df (pl.DataFrame): polars DataFrame

    Returns:
        pl.DataFrame: polars DataFrame
    """
    df = df.with_columns(
        pl.when(pl.col(col_name) == 0).then(2011).otherwise(2012).alias(col_name)
    )
    return df


def denormalise(df: pl.DataFrame, col_name: str, value: float) -> pl.DataFrame:
    """Denormalises a col_name by multiplying it by value.

    Args:
        df (pl.DataFrame): polars DataFrame

    Returns:
        pl.DataFrame: polars DataFrame
    """
    df = df.with_columns((pl.col(col_name) * value).alias(col_name))
    return df


def move_col(df: pl.DataFrame, col_name: str, position: int) -> pl.DataFrame:
    """Moves a column to a new position in the DataFrame.

    Args:
        df (pl.DataFrame): polars DataFrame
        col_name (str): column to be moved
        position (int): index position to which the column should be moved

    Returns:
        pl.DataFrame: polars DataFrame
    """
    cols = df.columns.copy()
    cols.remove(col_name)
    cols.insert(position, col_name)
    df = df.select(cols)
    return df


def check_sum(df: pl.DataFrame, sum_var: str, var1: str, var2: str):
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
