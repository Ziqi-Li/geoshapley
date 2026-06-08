"""Tree-path GeoShapley explainer.

This module implements an exact GeoShapley decomposition for tree models under
a tree-path-dependent value function. Missing players are integrated out using
the training/path cover proportions stored in each tree, in the same spirit as
TreeSHAP's ``tree_path_dependent`` setting.
"""

import itertools
import json
from types import SimpleNamespace

import numpy as np
import scipy.special

from .geoshapley import GeoShapleyResults


class GeoShapleyTreeExplainer:
    """Exact tree-path GeoShapley explainer for tree-based regressors.

    Parameters
    ----------
    model : object
        A fitted tree model. Currently supported model families are:

        * scikit-learn single regression trees exposing ``tree_``;
        * scikit-learn bagged/forest-style regressors exposing
          ``estimators_`` with fitted ``tree_`` objects, such as
          ``RandomForestRegressor`` and ``ExtraTreesRegressor``;
        * XGBoost sklearn regressors or native ``xgboost.Booster`` objects;
        * LightGBM sklearn regressors or native ``lightgbm.Booster`` objects;
        * FLAML ``AutoML`` objects whose selected estimator is one of the
          supported tree models.

    g : int, default=2
        Number of geographic coordinate columns. The last ``g`` columns in the
        explained data are treated as one grouped GEO player.

    model_output : {"raw"}, default="raw"
        Output space to explain. Only raw regression output is currently
        supported.

    Notes
    -----
    This explainer uses a different value function from
    :class:`GeoShapleyExplainer`. The Kernel explainer uses a user-supplied
    background dataset; this tree explainer uses tree path cover proportions.
    The two are often close when the background is the model training
    distribution, but they are not mathematically identical games.
    """

    def __init__(self, model, g=2, model_output="raw"):
        if model_output != "raw":
            raise ValueError("GeoShapleyTreeExplainer currently supports only model_output='raw'.")
        self.model = model
        self.g = g
        self.model_output = model_output
        self._adapter = None

    def explain(self, X_geo):
        """Explain observations and return a :class:`GeoShapleyResults`.

        Parameters
        ----------
        X_geo : pandas.DataFrame or numpy.ndarray
            Data to explain. Geographic coordinate columns must be the last
            ``g`` columns. Passing a DataFrame is recommended because
            ``GeoShapleyResults`` plotting and ``get_svc`` methods use column
            names and ``.values``.

        Returns
        -------
        GeoShapleyResults
            Standard GeoShapley result object with ``base_value``, ``primary``,
            ``geo`` and ``geo_intera`` attributes.
        """
        X_values = X_geo.values if hasattr(X_geo, "values") else np.asarray(X_geo)
        if X_values.ndim != 2:
            raise ValueError("X_geo must be a 2-dimensional array or DataFrame.")
        if X_values.shape[1] <= self.g:
            raise ValueError("X_geo must contain at least one non-geographic feature.")

        self.X_geo = X_geo
        self.M = X_values.shape[1]
        self.k = self.M - self.g
        self.background = None
        feature_names = list(X_geo.columns) if hasattr(X_geo, "columns") else None
        self._adapter = _make_tree_adapter(self.model, feature_names=feature_names)
        self.predict_f = self._adapter.predict
        self._precompute_design()

        values = np.vstack([self._explain_row(row) for row in X_values])
        primary = values[:, :self.k]
        geo = values[:, self.k]
        geo_intera = values[:, (self.k + 1):]
        return GeoShapleyResults(self, self.base_value, primary, geo, geo_intera)

    def _precompute_design(self):
        player_count = self.k + 1
        output_count = 2 * self.k + 1
        coalitions = [tuple(s) for s in _powerset(range(player_count))]
        self.coalitions = coalitions
        self.coalition_masks = np.array([
            sum(1 << player for player in coalition)
            for coalition in coalitions
        ], dtype=np.int64)

        Z = np.zeros((len(coalitions), output_count))
        kernels = np.zeros(len(coalitions))

        for i, coalition in enumerate(coalitions):
            if coalition:
                Z[i, coalition] = 1
            if self.k in coalition and len(coalition) > 1:
                for player in coalition:
                    if player < self.k:
                        Z[i, self.k + 1 + player] = 1
            kernels[i] = _shapley_kernel(player_count, len(coalition))

        ztw = Z.T * kernels
        self._projection = np.linalg.solve(ztw @ Z, ztw)
        self.base_value = self._adapter.empty_value()

    def _explain_row(self, row):
        coalition_values = self._adapter.coalition_values(row, self.k, self.coalition_masks)
        return self._projection @ (coalition_values - self.base_value)


class _PathEnsembleAdapter:
    """Adapter for additive ensembles represented as weighted path lists."""

    def __init__(self, trees, base_value=0.0):
        self.trees = trees
        self.base_value = float(base_value)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([
            self.full_value(row)
            for row in X
        ])

    def empty_value(self):
        return self.base_value + sum(
            tree["weight"] * _tree_empty_value(tree["paths"])
            for tree in self.trees
        )

    def full_value(self, row):
        return self.base_value + sum(
            tree["weight"] * _tree_full_value(tree["paths"], row)
            for tree in self.trees
        )

    def coalition_values(self, row, k, coalition_masks):
        values = np.full(len(coalition_masks), self.base_value)
        for tree in self.trees:
            values += tree["weight"] * _tree_coalition_values(
                tree["paths"],
                row,
                k,
                coalition_masks,
            )
        return values


def _make_tree_adapter(model, feature_names=None):
    model = _unwrap_flaml_model(model)
    if _is_xgboost_model(model):
        return _xgboost_adapter(model, feature_names=feature_names)
    if _is_lightgbm_model(model):
        return _lightgbm_adapter(model)
    return _sklearn_adapter(model)


def _unwrap_flaml_model(model):
    module = model.__class__.__module__
    if module.startswith("flaml.") and hasattr(model, "model"):
        flaml_model = model.model
        if hasattr(flaml_model, "estimator"):
            return flaml_model.estimator
        if hasattr(flaml_model, "_model"):
            return flaml_model._model
    if module.startswith("flaml.") and hasattr(model, "estimator"):
        return model.estimator
    return model


def _is_xgboost_model(model):
    return hasattr(model, "get_booster") or hasattr(model, "get_dump")


def _is_lightgbm_model(model):
    return hasattr(model, "booster_") or (
        hasattr(model, "dump_model") and model.__class__.__module__.startswith("lightgbm")
    )


def _sklearn_adapter(model):
    if hasattr(model, "tree_"):
        return _PathEnsembleAdapter([{
            "paths": _extract_sklearn_paths(model.tree_),
            "weight": 1.0,
        }])

    if hasattr(model, "estimators_"):
        if model.__class__.__name__ == "GradientBoostingRegressor":
            return _sklearn_gradient_boosting_adapter(model)

        trees = []
        estimators = list(np.ravel(model.estimators_))
        for estimator in estimators:
            if not hasattr(estimator, "tree_"):
                raise TypeError(
                    "Only sklearn ensembles whose estimators expose tree_ are supported."
                )
            trees.append({
                "paths": _extract_sklearn_paths(estimator.tree_),
                "weight": 1.0 / len(estimators),
            })
        return _PathEnsembleAdapter(trees)

    raise TypeError(
        "Unsupported model type. Expected a sklearn tree/forest-style regressor "
        "or an XGBoost regressor/booster."
    )


def _sklearn_gradient_boosting_adapter(model):
    if not hasattr(model, "learning_rate"):
        raise TypeError("GradientBoostingRegressor model is missing learning_rate.")

    if not hasattr(model, "init_") or not hasattr(model.init_, "constant_"):
        raise TypeError(
            "Only GradientBoostingRegressor models with a constant init_ are supported."
        )

    base_value = float(np.ravel(model.init_.constant_)[0])
    trees = []
    for estimator in np.ravel(model.estimators_):
        if not hasattr(estimator, "tree_"):
            raise TypeError("GradientBoostingRegressor estimators must expose tree_.")
        trees.append({
            "paths": _extract_sklearn_paths(estimator.tree_),
            "weight": float(model.learning_rate),
        })

    return _PathEnsembleAdapter(trees, base_value=base_value)


def _xgboost_adapter(model, feature_names=None):
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    feature_names = booster.feature_names or feature_names
    trees = [
        {"paths": _extract_xgboost_paths(json.loads(tree_json), feature_names), "weight": 1.0}
        for tree_json in booster.get_dump(with_stats=True, dump_format="json")
    ]
    return _PathEnsembleAdapter(trees, base_value=_xgboost_base_score(booster))


def _lightgbm_adapter(model):
    booster = model.booster_ if hasattr(model, "booster_") else model
    dump = booster.dump_model()
    trees = [
        {"paths": _extract_lightgbm_paths(tree["tree_structure"]), "weight": 1.0}
        for tree in dump["tree_info"]
    ]
    return _PathEnsembleAdapter(trees, base_value=0.0)


def _extract_sklearn_paths(tree):
    paths = []

    def walk(node_id, splits):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == right:
            paths.append({
                "splits": splits,
                "value": float(np.ravel(tree.value[node_id])[0]),
            })
            return

        total = tree.weighted_n_node_samples[node_id]
        left_probability = tree.weighted_n_node_samples[left] / total
        right_probability = tree.weighted_n_node_samples[right] / total

        feature = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        walk(left, splits + [_Split(feature, threshold, "le", left_probability)])
        walk(right, splits + [_Split(feature, threshold, "gt", right_probability)])

    walk(0, [])
    return paths


def _extract_xgboost_paths(root, feature_names=None):
    paths = []

    def walk(node, splits):
        if "leaf" in node:
            paths.append({
                "splits": splits,
                "value": float(node["leaf"]),
            })
            return

        cover = float(node["cover"])
        children = {child["nodeid"]: child for child in node["children"]}
        yes = children[node["yes"]]
        no = children[node["no"]]
        missing = children[node["missing"]]
        yes_probability = float(yes["cover"]) / cover if cover else 0.5
        no_probability = float(no["cover"]) / cover if cover else 0.5
        missing_probability = float(missing["cover"]) / cover if cover else 0.5
        feature = _feature_index(node["split"], feature_names, "XGBoost")
        threshold = float(node["split_condition"])

        # XGBoost uses float32 comparison internally for histogram/tree models.
        yes_direction = "lt_missing" if node["missing"] == node["yes"] else "lt"
        no_direction = "ge_missing" if node["missing"] == node["no"] else "ge"
        walk(yes, splits + [_Split(feature, threshold, yes_direction, yes_probability)])
        walk(no, splits + [_Split(feature, threshold, no_direction, no_probability)])
        if node["missing"] not in (node["yes"], node["no"]):
            walk(missing, splits + [_Split(feature, threshold, "missing", missing_probability)])

    walk(root, [])
    return paths


def _extract_lightgbm_paths(root):
    paths = []

    def walk(node, splits):
        if "leaf_value" in node:
            paths.append({
                "splits": splits,
                "value": float(node["leaf_value"]),
            })
            return

        decision_type = node.get("decision_type", "<=")
        if decision_type != "<=":
            raise ValueError(
                "GeoShapleyTreeExplainer currently supports only numerical "
                "LightGBM splits with decision_type '<='."
            )

        left = node["left_child"]
        right = node["right_child"]
        total = float(node.get("internal_count", 0.0))
        left_count = float(_lightgbm_node_count(left))
        right_count = float(_lightgbm_node_count(right))
        left_probability = left_count / total if total else 0.5
        right_probability = right_count / total if total else 0.5
        feature = int(node["split_feature"])
        threshold = float(node["threshold"])
        default_left = bool(node.get("default_left", True))

        left_direction = "le_missing" if default_left else "le"
        right_direction = "gt" if default_left else "gt_missing"
        walk(left, splits + [_Split(feature, threshold, left_direction, left_probability)])
        walk(right, splits + [_Split(feature, threshold, right_direction, right_probability)])

    walk(root, [])
    return paths


def _lightgbm_node_count(node):
    if "leaf_count" in node:
        return node["leaf_count"]
    return node["internal_count"]


def _tree_empty_value(paths):
    total = 0.0
    for path in paths:
        weight = 1.0
        for split in path["splits"]:
            weight *= split.probability
        total += path["value"] * weight
    return total


def _tree_full_value(paths, row):
    total = 0.0
    for path in paths:
        if all(split.matches(row) for split in path["splits"]):
            total += path["value"]
    return total


def _tree_coalition_values(paths, row, k, coalition_masks):
    values = np.zeros(len(coalition_masks))

    for path in paths:
        bad_mask = 0
        player_factors = {}
        base_weight = 1.0

        for split in path["splits"]:
            player = split.feature if split.feature < k else k
            if not split.matches(row):
                bad_mask |= 1 << player
            else:
                player_factors[player] = (
                    player_factors.get(player, 1.0) / split.probability
                )
            base_weight *= split.probability

        path_weights = np.full(len(coalition_masks), base_weight)
        path_weights[(coalition_masks & bad_mask) != 0] = 0.0

        for player, factor in player_factors.items():
            path_weights[(coalition_masks & (1 << player)) != 0] *= factor

        values += path["value"] * path_weights

    return values


class _Split(SimpleNamespace):
    def __init__(self, feature, threshold, direction, probability):
        if probability <= 0:
            raise ValueError("Tree path branch probabilities must be positive.")
        super().__init__(
            feature=int(feature),
            threshold=float(threshold),
            direction=direction,
            probability=float(probability),
        )

    def matches(self, row):
        value = row[self.feature]
        if self.direction == "missing":
            return np.isnan(value)
        if np.isnan(value):
            return False
        if self.direction == "le":
            return value <= self.threshold
        if self.direction == "gt":
            return value > self.threshold
        if self.direction == "le_missing":
            return np.isnan(value) or value <= self.threshold
        if self.direction == "gt_missing":
            return np.isnan(value) or value > self.threshold
        if self.direction == "lt":
            return np.float32(value) < np.float32(self.threshold)
        if self.direction == "ge":
            return np.float32(value) >= np.float32(self.threshold)
        if self.direction == "lt_missing":
            return np.isnan(value) or np.float32(value) < np.float32(self.threshold)
        if self.direction == "ge_missing":
            return np.isnan(value) or np.float32(value) >= np.float32(self.threshold)
        raise ValueError(f"Unknown split direction: {self.direction}")


def _feature_index(split_name, feature_names, model_name):
    if isinstance(split_name, int):
        return split_name
    if split_name.startswith("f") and split_name[1:].isdigit():
        return int(split_name[1:])
    if feature_names is not None and split_name in feature_names:
        return feature_names.index(split_name)
    raise ValueError(
        f"{model_name} split feature {split_name!r} could not be mapped to a "
        "column index. Pass X_geo as a DataFrame with matching column names or "
        "fit the model with unnamed array features."
    )


def _xgboost_base_score(booster):
    config = json.loads(booster.save_config())
    raw = config["learner"]["learner_model_param"].get("base_score", "0")
    if isinstance(raw, str):
        raw = raw.strip("[]")
    return float(raw)


def _powerset(iterable):
    items = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(items, r)
        for r in range(len(items) + 1)
    )


def _shapley_kernel(n_players, coalition_size):
    if coalition_size == 0 or coalition_size == n_players:
        return 100000000
    return (
        (n_players - 1)
        / (
            scipy.special.binom(n_players, coalition_size)
            * coalition_size
            * (n_players - coalition_size)
        )
    )
