from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def sample_split(
    df: pd.DataFrame,
    target_var: str,
    train_size: Optional[float] = 0.8,
    stratify: Optional[str] = None,
    n_bins: Optional[int] = 10,
) -> tuple:
    """Split the dataset into training and testing sets.

    Args:
        df_all (pd.DataFrame): DataFrame to split

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if stratify is not None:
        if df[stratify].nunique() > 31:  # Assuming 'cnt' is continuous
            df["bin"] = pd.qcut(df[stratify], q=n_bins, labels=False, duplicates="drop")
            stratify_col = df["bin"]
        else:
            stratify_col = df[target_var]

    predictors = [col for col in df.columns if col not in ["target_var", "bin"]]
    X_train, X_test, y_train, y_test = train_test_split(
        df[predictors],
        df[target_var],
        train_size=train_size,
        random_state=23,
        stratify=stratify_col if stratify is not None else None,
    )
    return X_train, X_test, y_train, y_test
