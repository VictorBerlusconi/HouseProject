from copy import deepcopy

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.evaluate import (
    aggregate_cv_metrics,
    iter_stratified_folds,
    log_saleprice_metrics,
    prefix_metrics,
    split_train_validation,
    summarize_model_results,
)
from src.features import build_impute_only_numeric_preprocessor, build_numeric_preprocessor


def get_training_target(dataset_data, target_strategy):
    """Build the target vector requested by a model candidate."""
    if target_strategy == "log_saleprice":
        return np.log(np.asarray(dataset_data["y"], dtype=float))

    if target_strategy == "log_price_per_sqft":
        if "GrLivArea" not in dataset_data["X"].columns:
            raise KeyError("GrLivArea is required for log_price_per_sqft target strategy.")
        gr_liv_area = dataset_data["X"]["GrLivArea"].to_numpy(dtype=float)
        if np.any(gr_liv_area <= 0):
            raise ValueError("GrLivArea must be strictly positive for log_price_per_sqft.")
        return np.log(np.asarray(dataset_data["y"], dtype=float) / gr_liv_area)

    raise ValueError(f"Unsupported target strategy: {target_strategy}")


def convert_predictions_to_log_saleprice(y_pred_target, X_features, target_strategy):
    """Convert any supported target prediction back to log SalePrice."""
    y_pred_target = np.asarray(y_pred_target, dtype=float)
    if target_strategy == "log_saleprice":
        return y_pred_target

    if target_strategy == "log_price_per_sqft":
        if "GrLivArea" not in X_features.columns:
            raise KeyError("GrLivArea is required to convert log_price_per_sqft predictions.")
        gr_liv_area = X_features["GrLivArea"].to_numpy(dtype=float)
        if np.any(gr_liv_area <= 0):
            raise ValueError("GrLivArea must be strictly positive to convert predictions.")
        return y_pred_target + np.log(gr_liv_area)

    raise ValueError(f"Unsupported target strategy: {target_strategy}")


def convert_predictions_to_saleprice(y_pred_target, X_features, target_strategy):
    """Convert model target predictions to original SalePrice values."""
    return np.exp(convert_predictions_to_log_saleprice(y_pred_target, X_features, target_strategy))


def build_estimator(model_key, params, seed):
    """Instantiate a configured estimator and inject reproducibility settings."""
    model_params = deepcopy(params)

    if model_key == "linear_regression":
        return LinearRegression(**model_params)
    if model_key == "lasso":
        return Lasso(**model_params)
    if model_key == "ridge":
        return Ridge(**model_params)
    if model_key == "knn":
        return KNeighborsRegressor(**model_params)
    if model_key == "decision_tree":
        model_params.setdefault("random_state", seed)
        return DecisionTreeRegressor(**model_params)
    if model_key == "random_forest":
        model_params.setdefault("random_state", seed)
        return RandomForestRegressor(**model_params)
    if model_key == "catboost":
        model_params.setdefault("random_seed", seed)
        model_params.setdefault("allow_writing_files", False)
        return CatBoostRegressor(**model_params)

    raise KeyError(f"Unknown model_key: {model_key}")


def build_preprocessor(dataset_data, preprocessor_kind):
    """Select the preprocessing pipeline requested by a candidate."""
    if preprocessor_kind == "scaled":
        return build_numeric_preprocessor(dataset_data)
    if preprocessor_kind == "impute_only":
        return build_impute_only_numeric_preprocessor(dataset_data)
    if preprocessor_kind is None:
        return None

    raise ValueError(f"Unsupported preprocessor_kind: {preprocessor_kind}")


def fit_predict_target(candidate, dataset_data, X_train, y_train, X_pred, seed):
    """Fit one candidate on a split and return predictions in its target space."""
    train_target = get_training_target({"X": X_train, "y": y_train}, candidate["target_strategy"])
    estimator = build_estimator(candidate["model_key"], candidate.get("params", {}), seed=seed)

    if candidate["model_key"] == "catboost":
        # CatBoost receives categorical feature names directly instead of sklearn preprocessing.
        estimator.fit(X_train, train_target, cat_features=dataset_data.get("cat_features", []))
        y_pred_target = estimator.predict(X_pred)
        return y_pred_target, estimator

    preprocessor = build_preprocessor(dataset_data, candidate["preprocessor_kind"])
    if preprocessor is None:
        model = estimator
    else:
        model = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )

    model.fit(X_train, train_target)
    y_pred_target = model.predict(X_pred)
    return y_pred_target, model


def train_and_evaluate_candidate(candidate, dataset_data, cv_config, seed):
    """Evaluate one candidate with inner CV and an outer validation holdout."""
    X_data = dataset_data["X"]
    y_data = dataset_data["y"]

    X_train_outer, X_val_outer, y_train_outer, y_val_outer = split_train_validation(
        X_data,
        y_data,
        validation_size=cv_config["validation_size"],
        random_state=seed,
        n_bins=cv_config["stratify_bins"],
    )

    fold_metrics = []
    # The inner folds estimate stability; the outer split is used for final candidate ranking.
    for fold, X_train, X_val, y_train, y_val in iter_stratified_folds(
        X_train_outer,
        y_train_outer,
        n_splits=cv_config["n_splits"],
        random_state=seed,
        n_bins=cv_config["stratify_bins"],
    ):
        y_pred_target, _ = fit_predict_target(
            candidate=candidate,
            dataset_data=dataset_data,
            X_train=X_train,
            y_train=y_train,
            X_pred=X_val,
            seed=seed + fold,
        )
        y_pred_log_saleprice = convert_predictions_to_log_saleprice(
            y_pred_target,
            X_features=X_val,
            target_strategy=candidate["target_strategy"],
        )
        fold_metrics.append(log_saleprice_metrics(y_val, y_pred_log_saleprice))

    y_pred_target, _ = fit_predict_target(
        candidate=candidate,
        dataset_data=dataset_data,
        X_train=X_train_outer,
        y_train=y_train_outer,
        X_pred=X_val_outer,
        seed=seed,
    )
    y_pred_log_saleprice = convert_predictions_to_log_saleprice(
        y_pred_target,
        X_features=X_val_outer,
        target_strategy=candidate["target_strategy"],
    )

    result_row = {
        "candidate_name": candidate["name"],
        "model_key": candidate["model_key"],
        "dataset_key": candidate["dataset_key"],
        "target_strategy": candidate["target_strategy"],
        **candidate.get("params", {}),
        **aggregate_cv_metrics(fold_metrics),
        **prefix_metrics(log_saleprice_metrics(y_val_outer, y_pred_log_saleprice), "val"),
    }
    return result_row


def evaluate_candidates(candidates, dataset_registry, cv_config, seed, sort_metric="val_rmse_log_saleprice"):
    """Evaluate all configured candidates and return a sorted comparison table."""
    result_rows = []
    for candidate in candidates:
        dataset_key = candidate["dataset_key"]
        if dataset_key not in dataset_registry:
            raise KeyError(f"Dataset key '{dataset_key}' is not available.")
        result_rows.append(
            train_and_evaluate_candidate(
                candidate=candidate,
                dataset_data=dataset_registry[dataset_key],
                cv_config=cv_config,
                seed=seed,
            )
        )

    return summarize_model_results(result_rows, sort_columns=(sort_metric,))


def select_best_candidate(results_df, candidates, sort_metric):
    """Find the best candidate config from the sorted results table."""
    best_result = results_df.sort_values(sort_metric).iloc[0]
    candidate_name = best_result["candidate_name"]
    candidate_lookup = {candidate["name"]: candidate for candidate in candidates}
    return candidate_lookup[candidate_name], best_result


def get_candidate_by_name(candidates, candidate_name):
    """Return a configured candidate by name."""
    candidate_lookup = {candidate["name"]: candidate for candidate in candidates}
    if candidate_name not in candidate_lookup:
        raise KeyError(f"Unknown candidate_name: {candidate_name}")
    return candidate_lookup[candidate_name]


def fit_full_candidate(candidate, train_dataset, seed):
    """Fit the selected candidate on the full training dataset."""
    train_target = get_training_target(train_dataset, candidate["target_strategy"])
    estimator = build_estimator(candidate["model_key"], candidate.get("params", {}), seed=seed)

    if candidate["model_key"] == "catboost":
        estimator.fit(train_dataset["X"], train_target, cat_features=train_dataset.get("cat_features", []))
        return {
            "family": "catboost",
            "model": estimator,
            "candidate": candidate,
        }

    preprocessor = build_preprocessor(train_dataset, candidate["preprocessor_kind"])
    if preprocessor is None:
        model = estimator
    else:
        model = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
    model.fit(train_dataset["X"], train_target)
    return {
        "family": "sklearn",
        "model": model,
        "candidate": candidate,
    }
