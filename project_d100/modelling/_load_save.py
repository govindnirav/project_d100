import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

MODEL = Path(__file__).parent.parent.parent / "models"
SPLIT = Path(__file__).parent.parent.parent / "data" / "processed"


def save_model(pipeline: Pipeline, name: str) -> None:
    """Saves best (tuned) pipeline as .pkl file with name `name`.
    Name is cleaned to remove illegal characters and file extensions.

    Args:
        pipeline (Pipeline): best pipeline
        name (str): name of the file
    """
    clean_name = re.sub(
        r"[^\w\-_\. ]", "_", name
    )  # Replace illegal characters with underscores
    clean_name = Path(name).stem  # Removes any file extension
    joblib.dump(pipeline, MODEL / f"{clean_name}.pkl")


def load_model(name: str) -> Pipeline:
    """Loads a model from the models directory.

    Args:
        name (str): name of the model

    Returns:
        Pipeline: model pipeline
    """
    return joblib.load(MODEL / f"{name}.pkl")


def save_split(
    X_train: pd.DataFrame, X_test: pd.DataFggrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    """Save the train and test sets as .pkl files

    Args:
        X_train (pd.DataFrame): training features
        X_test (pd.DataFrame): testing features
        y_train (pd.Series): training target
        y_test (pd.Series): testing target
    """
    joblib.dump((X_train, y_train), SPLIT / "train_set.pkl")
    joblib.dump((X_test, y_test), SPLIT / "test_set.pkl")


def load_split() -> tuple:
    """Load the train and test sets from the .pkl files

    Returns:
        Tuple: train and test sets
    """
    X_train, y_train = joblib.load(SPLIT / "train_set.pkl")
    X_test, y_test = joblib.load(SPLIT / "test_set.pkl")
    return X_train, X_test, y_train, y_test
