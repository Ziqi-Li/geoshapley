import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from geoshapley import GeoShapleyTreeExplainer


pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names.*:UserWarning"
)


def _toy_data(n=80, seed=1, g=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2 + g))
    y = (
        2.0 * X[:, 0]
        - 1.0 * X[:, 1]
        + 0.5 * X[:, -1]
        + X[:, 0] * X[:, -2]
    )
    columns = ["x1", "x2"] + [f"coord{i + 1}" for i in range(g)]
    return pd.DataFrame(X, columns=columns), y


def _assert_additive(model, X, result, atol=1e-8):
    total = (
        result.base_value
        + result.primary.sum(axis=1)
        + result.geo
        + result.geo_intera.sum(axis=1)
    )
    np.testing.assert_allclose(total, model.predict(X.values), atol=atol)


def test_decision_tree_additivity():
    X, y = _toy_data()
    model = DecisionTreeRegressor(max_depth=4, random_state=1).fit(X.values, y)

    result = GeoShapleyTreeExplainer(model, g=2).explain(X)

    assert result.primary.shape == (len(X), 2)
    assert result.geo.shape == (len(X),)
    assert result.geo_intera.shape == (len(X), 2)
    _assert_additive(model, X, result)


def test_random_forest_additivity():
    X, y = _toy_data()
    model = RandomForestRegressor(
        n_estimators=5,
        max_depth=4,
        random_state=1,
        n_jobs=1,
    ).fit(X.values, y)

    result = GeoShapleyTreeExplainer(model, g=2).explain(X)

    _assert_additive(model, X, result)


def test_gradient_boosting_additivity():
    X, y = _toy_data()
    model = GradientBoostingRegressor(
        n_estimators=8,
        max_depth=3,
        learning_rate=0.1,
        random_state=1,
    ).fit(X.values, y)

    result = GeoShapleyTreeExplainer(model, g=2).explain(X)

    _assert_additive(model, X, result)


def test_g1_matches_tree_shap_after_redistribution():
    shap = pytest.importorskip("shap")
    X, y = _toy_data(g=1)
    model = RandomForestRegressor(
        n_estimators=5,
        max_depth=4,
        random_state=1,
        n_jobs=1,
    ).fit(X.values, y)

    result = GeoShapleyTreeExplainer(model, g=1).explain(X)
    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(X)
    redistributed = result.geoshap_to_shap()
    expected_value = np.ravel(tree_explainer.expected_value)[0]

    np.testing.assert_allclose(result.base_value, expected_value)
    np.testing.assert_allclose(redistributed, shap_values, atol=1e-7)
    np.testing.assert_allclose(
        result.base_value + redistributed.sum(axis=1),
        model.predict(X.values),
        atol=1e-7,
    )


def test_xgboost_additivity_if_available():
    xgboost = pytest.importorskip("xgboost")
    X, y = _toy_data()
    model = xgboost.XGBRegressor(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=1,
        n_jobs=1,
    ).fit(X.values, y)

    result = GeoShapleyTreeExplainer(model, g=2).explain(X)

    _assert_additive(model, X, result, atol=1e-5)


def test_native_xgboost_booster_additivity_if_available():
    xgboost = pytest.importorskip("xgboost")
    X, y = _toy_data()
    dtrain = xgboost.DMatrix(X.values, label=y)
    booster = xgboost.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "eta": 0.1,
            "seed": 1,
            "nthread": 1,
        },
        dtrain,
        num_boost_round=5,
    )

    result = GeoShapleyTreeExplainer(booster, g=2).explain(X)
    total = (
        result.base_value
        + result.primary.sum(axis=1)
        + result.geo
        + result.geo_intera.sum(axis=1)
    )

    np.testing.assert_allclose(total, booster.predict(xgboost.DMatrix(X.values)), atol=1e-5)


def test_xgboost_named_features_additivity_if_available():
    xgboost = pytest.importorskip("xgboost")
    X, y = _toy_data()
    model = xgboost.XGBRegressor(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=1,
        n_jobs=1,
    ).fit(X, y)

    result = GeoShapleyTreeExplainer(model, g=2).explain(X)

    _assert_additive(model, X, result, atol=1e-5)


def test_lightgbm_additivity_if_available():
    lightgbm = pytest.importorskip("lightgbm")
    X, y = _toy_data()
    model = lightgbm.LGBMRegressor(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        min_child_samples=2,
        random_state=1,
        n_jobs=1,
        verbose=-1,
    ).fit(X, y)

    result = GeoShapleyTreeExplainer(model, g=2).explain(X)

    _assert_additive(model, X, result, atol=1e-8)


def test_native_lightgbm_booster_additivity_if_available():
    lightgbm = pytest.importorskip("lightgbm")
    X, y = _toy_data()
    dataset = lightgbm.Dataset(X.values, label=y)
    booster = lightgbm.train(
        {
            "objective": "regression",
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_data_in_leaf": 2,
            "num_threads": 1,
            "verbose": -1,
            "seed": 1,
        },
        dataset,
        num_boost_round=5,
    )

    result = GeoShapleyTreeExplainer(booster, g=2).explain(X)
    total = (
        result.base_value
        + result.primary.sum(axis=1)
        + result.geo
        + result.geo_intera.sum(axis=1)
    )

    np.testing.assert_allclose(total, booster.predict(X.values), atol=1e-8)


def test_flaml_automl_xgboost_model_additivity_if_available():
    flaml = pytest.importorskip("flaml")
    X, y = _toy_data()
    automl = flaml.AutoML()
    automl.fit(
        X_train=X.values,
        y_train=y,
        task="regression",
        estimator_list=["xgboost"],
        time_budget=3,
        n_jobs=1,
        verbose=0,
    )

    result = GeoShapleyTreeExplainer(automl, g=2).explain(X)
    total = (
        result.base_value
        + result.primary.sum(axis=1)
        + result.geo
        + result.geo_intera.sum(axis=1)
    )

    np.testing.assert_allclose(total, automl.predict(X.values), atol=1e-5)


def test_flaml_automl_lightgbm_model_additivity_if_available():
    flaml = pytest.importorskip("flaml")
    pytest.importorskip("lightgbm")
    X, y = _toy_data()
    automl = flaml.AutoML()
    automl.fit(
        X_train=X.values,
        y_train=y,
        task="regression",
        estimator_list=["lgbm"],
        time_budget=3,
        n_jobs=1,
        verbose=0,
    )

    result = GeoShapleyTreeExplainer(automl, g=2).explain(X)
    total = (
        result.base_value
        + result.primary.sum(axis=1)
        + result.geo
        + result.geo_intera.sum(axis=1)
    )

    np.testing.assert_allclose(total, automl.predict(X.values), atol=1e-8)
