from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def pipeline_preprocessing(
    numericals: list[str], categoricals: list[str]
) -> ColumnTransformer:
    """Create a pipeline for preprocessing data.

    Args:
    numericals (list[str]): list of numerical columns
    categoricals (list[str]): list of categorical columns

    Returns:
        ColumnTransformer: preprocessing pipeline
    """
    # Numerical pipeline
    num_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("spline", SplineTransformer(include_bias=False, knots="quantile")),
            # Intercept not transformed; knots at quantiles (not uniformly spaced)
        ]
    )

    # Categorical pipeline
    cat_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(sparse_output=False, drop="first")),
        ]
    )

    # Full pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numericals),
            ("cat", cat_pipeline, categoricals),
        ]
    )

    return preprocessor
