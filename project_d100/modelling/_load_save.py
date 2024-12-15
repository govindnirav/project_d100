import re
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

MODEL = Path(__file__).parent.parent.parent / "models"


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
