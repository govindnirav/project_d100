import re
from pathlib import Path

import polars as pl

CLEAN_DATA = Path(__file__).parent.parent.parent / "data" / "clean"


def save_parquet(df: pl.DataFrame, name: str):
    """Saves a polars DataFrame as a parquet file with name `name`.
    Name is cleaned to remove illegal characters and file extensions.

    Args:
        df (pl.DataFrame): polars DataFrame
        name (str): name of the file
    """
    clean_name = re.sub(
        r"[^\w\-_\. ]", "_", name
    )  # Replace illegal characters with underscores
    clean_name = Path(name).stem  # Removes any file extension
    df.write_parquet(CLEAN_DATA / f"{clean_name}.parquet")
