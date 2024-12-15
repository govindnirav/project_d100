from typing import Optional

import numpy as np
import pandas as pd
from glum import TweedieDistribution
from sklearn.metrics import auc
from sklearn.pipeline import Pipeline

from project_d100.preprocessing import sample_split


def evaluate_predictions(
    df: pd.DataFrame,
    target_var: str,
    pipeline: Pipeline,
    train_size: Optional[float] = 0.8,
    stratify: Optional[str] = None,
    n_bins: Optional[int] = 10,
):
    """Evaluate predictions against actual outcomes.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe used for evaluation
    outcome_column : str
        Name of outcome column
    preds_column : str, optional
        Name of predictions column, by default None
    model :
        Fitted model, by default None
    tweedie_power : float, optional
        Power of tweedie distribution for deviance computation, by default 1.5
    exposure_column : str, optional
        Name of exposure column, by default None

    Returns
    -------
    evals
        DataFrame containing metrics
    """

    evals = {}

    X_train, X_test, y_train, y_test = sample_split(
        df,
        target_var=target_var,
        train_size=train_size,
        stratify=stratify,
        n_bins=n_bins,
    )

    preds = pipeline.predict(X_test)

    evals["mean_preds"] = np.average(preds)
    evals["mean_outcome"] = np.average(y_test)
    evals["bias"] = (evals["mean_preds"] - evals["mean_outcome"]) / evals[
        "mean_outcome"
    ]

    evals["mse"] = np.average((preds - y_test) ** 2)
    evals["rmse"] = np.sqrt(evals["mse"])
    evals["mae"] = np.average(np.abs(preds - y_test))
    evals["deviance"] = TweedieDistribution(1.5).deviance(y_test, preds)
    ordered_samples, cum_actuals = _lorenz_curve(y_test, preds)
    evals["gini"] = 1 - 2 * auc(ordered_samples, cum_actuals)

    return pd.DataFrame(evals, index=[0]).T


def _lorenz_curve(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_y_true = y_true[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_cnt = np.cumsum(ranked_pure_premium * ranked_y_true)
    cumulated_cnt = cumulated_cnt.astype(float)
    cumulated_cnt /= cumulated_cnt[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_cnt))
    return cumulated_samples, cumulated_cnt
