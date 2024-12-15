from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from project_d100.preprocessing import _pipeline_preprocessing


def lgbm_pipeline(numericals: list[str], categoricals: list[str]) -> Pipeline:
    """Create a pipeline for a LGBM model

    Args:
        numericals (list[str]): list of numerical columns
        categoricals (list[str]): list of categorical columns

    Returns:
        Pipeline: LGBM pipeline
    """
    # Preprocessing pipeline
    preprocessor = _pipeline_preprocessing(numericals, categoricals)
    preprocessor.set_output(transform="pandas")
    # Confirms output is a pandas DataFrame

    # LGBM model
    # Using Tweedie distribution because target is zero-inflated
    # Uses a tweedie loss with power 1.5
    model = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5)
    # L1 ratio set to 1 - some features selection occurs
    # Model searches for optimal alpha

    # Full pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline
