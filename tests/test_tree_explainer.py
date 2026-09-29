import itertools
import math

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


def _exhaustive_projection(explainer, row):
    """Reference implementation of the original exponential algorithm."""
    k = explainer.k
    player_count = k + 1
    coalitions = list(itertools.chain.from_iterable(
        itertools.combinations(range(player_count), size)
        for size in range(player_count + 1)
    ))
    masks = np.array([
        sum(1 << player for player in coalition)
        for coalition in coalitions
    ])
    design = np.zeros((len(coalitions), 2 * k + 1))
    weights = np.zeros(len(coalitions))

    for i, coalition in enumerate(coalitions):
        if coalition:
            design[i, list(coalition)] = 1.0
        if k in coalition:
            for player in coalition:
                if player < k:
                    design[i, k + 1 + player] = 1.0
        size = len(coalition)
        weights[i] = (
            1e8
            if size in (0, player_count)
            else (player_count - 1)
            / (math.comb(player_count, size) * size * (player_count - size))
        )

    coalition_values = np.full(len(coalitions), explainer._adapter.base_value)
    for tree in explainer._adapter.trees:
        tree_values = np.zeros(len(coalitions))
        for path in tree["paths"]:
            base_weight = 1.0
            bad_mask = 0
            factors = {}
            for split in path["splits"]:
                player = split.feature if split.feature < k else k
                base_weight *= split.probability
                if split.matches(row):
                    factors[player] = factors.get(player, 1.0) / split.probability
                else:
                    bad_mask |= 1 << player

            path_weights = np.full(len(coalitions), base_weight)
            path_weights[(masks & bad_mask) != 0] = 0.0
            for player, factor in factors.items():
                path_weights[(masks & (1 << player)) != 0] *= factor
            tree_values += path["value"] * path_weights
        coalition_values += tree["weight"] * tree_values

    weighted_design = design.T * weights
    projection = np.linalg.solve(
        weighted_design @ design,
        weighted_design,
    )
    return projection @ (coalition_values - explainer.base_value)


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


def test_polynomial_algorithm_matches_exhaustive_projection():
    X, y = _toy_data(n=50, seed=4, g=2)
    model = RandomForestRegressor(
        n_estimators=3,
        max_depth=4,
        random_state=3,
        n_jobs=1,
    ).fit(X.values, y)
    explained = X.iloc[:4]

    explainer = GeoShapleyTreeExplainer(model, g=2)
    result = explainer.explain(explained)
    actual = np.column_stack((result.primary, result.geo, result.geo_intera))
    expected = np.vstack([
        _exhaustive_projection(explainer, row)
        for row in explained.values
    ])

    np.testing.assert_allclose(actual, expected, atol=5e-7)


def test_many_features_do_not_require_coalition_enumeration():
    rng = np.random.default_rng(8)
    X_values = rng.normal(size=(100, 82))
    y = X_values[:, 0] * X_values[:, 80] + X_values[:, 1] - X_values[:, 81]
    columns = [f"x{i}" for i in range(80)] + ["lat", "lon"]
    X = pd.DataFrame(X_values, columns=columns)
    model = DecisionTreeRegressor(max_depth=4, random_state=2).fit(X.values, y)

    result = GeoShapleyTreeExplainer(model, g=2).explain(X.iloc[:3])

    assert result.primary.shape == (3, 80)
    assert result.geo_intera.shape == (3, 80)
    _assert_additive(model, X.iloc[:3], result)


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
