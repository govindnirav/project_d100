from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, SplineTransformer


def _pipeline_preprocessing(
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
            ("scaler", RobustScaler(quantile_range=(1.0, 99.0))),
            # Ensures no feature dominates the objective funtion
            # Scales robustly to outliers (unlike StandardScaler)
            ("spline", SplineTransformer(include_bias=False, knots="quantile")),
            # Intercept not transformed; knots at quantiles (not uniformly spaced)
            # Prevents overfitting by allowing non-linear relationships
            # Has continuitiy (unlike binning)
        ]
    )

    # Categorical pipeline
    cat_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    sparse_output=False,
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                ),
            ),
            # Not necessary if using LightGBM only, but also using GLM
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
