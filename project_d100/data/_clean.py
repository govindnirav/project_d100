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


def denormalise_temp(
    df: pl.DataFrame, col_name: str, t_max: float, t_min: float
) -> pl.DataFrame:
    """Denormalises temperature by multiplying it by a function of t_max and t_min
        (See additional_info.txt for more details)

    Args:
        df (pl.DataFrame): polars DataFrame
        col_name (str): column to be denormalised
        t_max (float): maximum temperature
        t_min (float): minimum temperature

    Returns:
        pl.DataFrame: polars DataFrame
    """
    df = df.with_columns(
        (((pl.col(col_name) * (t_max - t_min)) + t_min)).alias(col_name)
    )
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


def log_transform(df: pl.DataFrame, var: str) -> pl.DataFrame:
    """Applies the natural log transformation to a variable in a polars DataFrame.

    Args:
        df (pl.DataFrame): polars DataFrame
        var (str): name of varaible to be transformed

    Returns:
        pl.DataFrame: polars DataFrame
    """
    df = df.with_columns((pl.col(var).log10()).alias(f"log_{var}"))
    return df
