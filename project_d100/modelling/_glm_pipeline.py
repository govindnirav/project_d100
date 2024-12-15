from glum import GeneralizedLinearRegressor, TweedieDistribution
from sklearn.pipeline import Pipeline

from project_d100.preprocessing import _pipeline_preprocessing


def glm_pipeline(numericals: list[str], categoricals: list[str]) -> Pipeline:
    """Create a pipeline for a GLM model

    Args:
        numericals (list[str]): list of numerical columns
        categoricals (list[str]): list of categorical columns

    Returns:
        Pipeline: GLM pipeline
    """
    # Preprocessing pipeline
    preprocessor = _pipeline_preprocessing(numericals, categoricals)
    preprocessor.set_output(transform="pandas")
    # Confirms output is a pandas DataFrame

    # GLM model
    # Using Tweedie distribution because target is zero-inflated
    TweedieDist = TweedieDistribution(1.5)
    # Uses a tweedie loss with power 1.5
    model = GeneralizedLinearRegressor(family=TweedieDist, fit_intercept=True)
    # L1 ratio set to 1 - some features selection occurs
    # Model searches for optimal alpha

    # Full pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline
