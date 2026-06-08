import numpy as np
import pandas as pd

from geoshapley import GeoShapleyExplainer


def test_kernel_explainer_linear_model_matches_expected_effects():
    X = pd.DataFrame({
        "x1": [-1.0, 0.0, 1.0, 2.0],
        "x2": [2.0, 0.0, -2.0, 1.0],
        "geo": [0.0, 1.0, 2.0, 3.0],
    })
    background = X.values

    def predict(values):
        values = np.asarray(values)
        return 1.0 + 2.0 * values[:, 0] - 3.0 * values[:, 1] + 5.0 * values[:, 2]

    result = GeoShapleyExplainer(
        predict,
        background=background,
        g=1,
        exact=True,
    ).explain(X, n_jobs=1)

    expected_base = predict(background).mean()
    expected_primary = np.column_stack([
        2.0 * (X["x1"].values - X["x1"].mean()),
        -3.0 * (X["x2"].values - X["x2"].mean()),
    ])
    expected_geo = 5.0 * (X["geo"].values - X["geo"].mean())

    np.testing.assert_allclose(result.base_value, expected_base)
    np.testing.assert_allclose(result.primary, expected_primary, atol=1e-5)
    np.testing.assert_allclose(result.geo, expected_geo, atol=1e-5)
    np.testing.assert_allclose(result.geo_intera, 0.0, atol=1e-5)


def test_kernel_explainer_is_additive_for_nonlinear_model():
    X = pd.DataFrame({
        "x1": [-1.0, 0.0, 1.0, 2.0],
        "x2": [2.0, 0.0, -2.0, 1.0],
        "lat": [0.0, 1.0, 2.0, 3.0],
        "lon": [1.0, 1.5, 2.0, 2.5],
    })

    def predict(values):
        values = np.asarray(values)
        return (
            values[:, 0] ** 2
            - values[:, 1]
            + 0.5 * values[:, 2]
            + values[:, 0] * values[:, 3]
        )

    result = GeoShapleyExplainer(
        predict,
        background=X.values,
        g=2,
        exact=True,
    ).explain(X, n_jobs=1)

    total = (
        result.base_value
        + result.primary.sum(axis=1)
        + result.geo
        + result.geo_intera.sum(axis=1)
    )
    np.testing.assert_allclose(total, predict(X.values), atol=1e-5)
    assert result.primary.shape == (len(X), 2)
    assert result.geo.shape == (len(X),)
    assert result.geo_intera.shape == (len(X), 2)
