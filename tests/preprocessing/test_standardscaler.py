import numpy as np
import pandas as pd
import pytest

from project_d100.preprocessing import StandardScaler


@pytest.mark.parametrize(
    "X",
    [
        np.array([12, 52, 233, 34, 4235, 2346, 723, 448, 9234234, 1230, 121, 14322]),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        np.array([[0.1, 0.5, 0.3], [0.4, 0.5, 0.6], [0.7, 0.5, 0.9]]),
        pd.DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]}, index=[0, 1, 2, 3, 4]
        ),
    ],
)
# Check mean is scaled to 0 and st.dev is scaled to 1
def test_mean_var(X):
    Xt = StandardScaler().fit_transform(X)

    mean = np.mean(Xt, axis=0)
    std = np.std(Xt, axis=0)

    assert (np.all(np.isclose(mean, 0))) & (
        np.all(np.isclose(std, 1) | (np.isclose(std, 0)))
    )
