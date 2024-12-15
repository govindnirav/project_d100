from feature_engine.outliers import Winsorizer
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
            (
                "winsorizer",
                Winsorizer(capping_method="quantiles", tail="both", fold=0.01),
            ),
            # Prevents outliers from dominating the objective function
            # Before scaling, to prevent outliers from affecting the mean and variance
            ("scaler", StandardScaler()),
            # Ensures no feature dominates the objective funtion
            ("spline", SplineTransformer(include_bias=False, knots="quantile")),
            # Intercept not transformed; knots at quantiles (not uniformly spaced)
            # Prevents overfitting by allowing non-linear relationships
            # Has continuitiy (unlike binning)
        ]
    )

    # Categorical pipeline
    cat_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(drop="first")),
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
