from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


def glm_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline,
    param_grid: Optional[dict[str, list]] = None,
    search: Optional[str] = "random",
    n_iter: Optional[int] = 20,
) -> Pipeline:
    """Tune a GLM model using cv search

    Args:
        X_train (pd.DataFrame): training data
        y_train (pd.Series): training target
        pipeline (Pipeline): GLM pipeline
        param_grid (Optional[Union[dict, dict[list]]], optional): hyperparameter grid
        search (Optional[str], optional): search method
        n_iter (Optional[int], optional): number of iterations for random search

    Returns:
        Pipeline: tuned GLM pipeline
    """
    # Default hyperparameter grid
    if param_grid is None:
        param_grid = {
            "model__alpha": np.logspace(-3, 3, 7),
            # logspace returns numbers spaced evenly on a log scale
            # between 10^-3 and 10^3
            # deals with small and large values better than a linear scale
            "model__l1_ratio": [0, 0.25, 0.5, 0.75, 1.0],
        }

    # Search method
    if search == "grid":
        cv = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            n_jobs=1,
            scoring="neg_mean_squared_error",
        )
    elif search == "random":
        cv = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=5,
            n_jobs=1,
            scoring="neg_mean_squared_error",
            random_state=42,
            verbose=2,
        )

    # Fit model
    cv.fit(X_train, y_train)

    # Best pipeline after tuning
    best_pipeline = cv.best_estimator_

    print("\nBest parameters:\n", cv.best_params_)
    print("Best score (GLM):", cv.best_score_)

    return best_pipeline


def lgbm_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline,
    param_grid: Optional[dict[str, list]] = None,
    search: Optional[str] = "random",
    n_iter: Optional[int] = 20,
) -> Pipeline:
    """Tune a LGBM model using cv search

    Args:
        X_train (pd.DataFrame): training data
        y_train (pd.Series): training target
        pipeline (Pipeline): GLM pipeline
        param_grid (Optional[Union[dict, dict[list]]], optional): hyperparameter grid
        search (Optional[str], optional): search method
        n_iter (Optional[int], optional): number of iterations for random search

    Returns:
        Pipeline: tuned LGBM pipeline
    """
    # Default hyperparameter grid
    if param_grid is None:
        param_grid = {
            "model__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
            "model__n_estimators": [50, 100, 150, 200],
            "model__n_leaves": [31, 63, 127, 255],
            "model__min_child_weight": [1, 2, 5, 10],
        }

    # Search method
    if search == "grid":
        cv = GridSearchCV(
            pipeline, param_grid, cv=5, n_jobs=1, scoring="neg_mean_squared_error"
        )
    elif search == "random":
        cv = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=5,
            n_jobs=1,
            scoring="neg_mean_squared_error",
            random_state=42,
            verbose=2,
        )

    # Fit model
    cv.fit(X_train, y_train)

    # Best pipeline after tuning
    best_pipeline = cv.best_estimator_

    print("\nBest parameters:\n", cv.best_params_)
    print("Best score (LGBM):", cv.best_score_)

    return best_pipeline
