import numpy as np
import pandas as pd
from glum import TweedieDistribution
from sklearn.metrics import auc
from sklearn.pipeline import Pipeline


def evaluate_predictions(
    df: pd.DataFrame,
    target_var: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Evaluate predictions against actual outcomes.
        Uses the following metrics:
        - Mean prediction
        - Mean outcome
        - Bias (scaled)
        - Mean squared error
        - Root mean squared error
        - Mean absolute error
        - Deviance
        - Gini coefficient

    Args:
        df (pd.DataFrame): pandas DataFrame containing the target variable
        target_var (str): target variable
        pipeline (Pipeline): tuned model pipeline
        train_size (float): proportion of data to use for training
        stratify (str): column to stratify the training split
        n_bins (int): number of bins to use for stratification

    Returns:
        pd.DataFrame: evaluation metrics
        pd.Series: actual outcomes
        pd.Series: predicted outcomes
    """

    evals = {}

    y_preds = pipeline.predict(X_test)

    evals["mean_preds"] = np.average(y_preds)
    evals["mean_outcome"] = np.average(y_test)
    evals["bias"] = (evals["mean_preds"] - evals["mean_outcome"]) / evals[
        "mean_outcome"
    ]

    evals["mse"] = np.average((y_preds - y_test) ** 2)
    evals["rmse"] = np.sqrt(evals["mse"])
    evals["mae"] = np.average(np.abs(y_preds - y_test))
    evals["deviance"] = TweedieDistribution(1.5).deviance(y_test, y_preds)
    ordered_samples, cum_actuals = _calculate_lorenz(y_test, y_preds)
    oracle_samples, oracle_actuals = _calculate_lorenz(y_test, y_test)
    oracle_gini = 1 - 2 * auc(oracle_samples, oracle_actuals)
    evals["gini"] = 1 - 2 * auc(ordered_samples, cum_actuals)
    evals["normalised_gini"] = evals["gini"] / oracle_gini

    return pd.DataFrame(evals, index=[0]).T, y_test, y_preds


def _calculate_lorenz(y_true, y_preds):
    """Calculate the Lorenz curve coordinates

    Args:
        y_true (np.ndarray | pd.Series): actual outcomes
        y_preds (np.ndarray | pd.Series): predicted outcomes

    Returns:
        tuple: cumulative share of samples (x-axis),
               cumulative share of actuals (y-axis)
    """
    y_true, y_preds = np.asarray(y_true), np.asarray(y_preds)

    ranking = np.argsort(y_preds)
    ranked_y_true = y_true[ranking]

    cumulated_cnt = np.cumsum(ranked_y_true)
    cumulated_cnt = cumulated_cnt.astype(float)
    cumulated_cnt /= cumulated_cnt[-1]  # Normalize to 1
    cumulated_samples = np.linspace(0, 1, len(cumulated_cnt))

    return cumulated_samples, cumulated_cnt
