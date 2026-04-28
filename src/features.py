import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Missing values in these columns mean absence of a house feature, not unknown data.
MISSING_CATEGORY_LABELS = {
    "Alley": "NoAlleyAccess",
    "BsmtQual": "NoBasement",
    "BsmtCond": "NoBasement",
    "BsmtExposure": "NoBasement",
    "BsmtFinType1": "NoBasement",
    "BsmtFinType2": "NoBasement",
    "FireplaceQu": "NoFireplace",
    "GarageType": "NoGarage",
    "GarageFinish": "NoGarage",
    "GarageQual": "NoGarage",
    "GarageCond": "NoGarage",
    "PoolQC": "NoPool",
    "Fence": "NoFence",
    "MiscFeature": "None",
}

# Ordered categorical values are converted to numeric rank before modeling.
ORDINAL_MAPPINGS = {
    "LotShape": {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1},
    "Utilities": {"AllPub": 4, "NoSewr": 3, "NoSeWa": 2, "ELO": 1},
    "LandSlope": {"Gtl": 3, "Mod": 2, "Sev": 1},
    "ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    "ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NoBasement": 0},
    "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NoBasement": 0},
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NoBasement": 0},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NoBasement": 0},
    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NoBasement": 0},
    "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    "CentralAir": {"Y": 1, "N": 0},
    "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
    "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NoFireplace": 0},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "NoGarage": 0},
    "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NoGarage": 0},
    "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NoGarage": 0},
    "PavedDrive": {"Y": 2, "P": 1, "N": 0},
    "PoolQC": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "NoPool": 0},
    "Fence": {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "NoFence": 0},
}


def prepare_common_data(df_input):
    """Apply shared missing-value handling and ordinal encoding."""
    df_prepared = df_input.copy()

    for column, fill_value in MISSING_CATEGORY_LABELS.items():
        if column in df_prepared.columns:
            df_prepared[column] = df_prepared[column].fillna(fill_value)

    for column, mapping in ORDINAL_MAPPINGS.items():
        if column in df_prepared.columns:
            df_prepared[column] = df_prepared[column].map(mapping)

    return df_prepared


def split_X_y(df_input, target_column, id_column):
    """Split a competition dataframe into model features and target."""
    X = df_input.drop(columns=[target_column, id_column], errors="ignore").copy()
    y = df_input[target_column].copy()
    return X, y


def log_target(y):
    """Return a log-transformed target series with index preserved."""
    y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y)
    y_series = y_series.astype(float)
    target_name = y_series.name or "target"
    return pd.Series(np.log(y_series), index=y_series.index, name=f"log_{target_name}")


def categorical_columns(df_input):
    """Return columns that should be treated as categorical."""
    return df_input.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def prepare_onehot_features(X_data):
    """One-hot encode remaining categorical columns for sklearn estimators."""
    X_encoded = X_data.copy()
    onehot_categorical_columns = categorical_columns(X_encoded)

    for column in onehot_categorical_columns:
        X_encoded[column] = X_encoded[column].fillna("Missing").astype(str)

    if onehot_categorical_columns:
        X_encoded = pd.get_dummies(
            X_encoded,
            columns=onehot_categorical_columns,
            dtype=float,
        )

    return X_encoded


def get_ordinal_columns(X_data):
    """Return ordinal-encoded columns present in the given feature frame."""
    return [column for column in ORDINAL_MAPPINGS if column in X_data.columns]


def get_scale_columns(X_data):
    """Return non-ordinal numeric columns that are safe to scale."""
    ordinal_columns = set(get_ordinal_columns(X_data))
    return [column for column in X_data.columns if column not in ordinal_columns]


def build_numeric_preprocessor(dataset_data):
    """Build imputation and scaling preprocessing for linear/distance models."""
    return ColumnTransformer(
        transformers=[
            (
                "scaled_num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                dataset_data.get("scale_columns", []),
            ),
            (
                "ordinal_passthrough",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                dataset_data.get("ordinal_columns", []),
            ),
            (
                "non_scaled_passthrough",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                dataset_data.get("non_scaled_columns", []),
            ),
        ],
        remainder="drop",
    )


def build_impute_only_numeric_preprocessor(dataset_data):
    """Build imputation-only preprocessing for tree-based sklearn models."""
    return ColumnTransformer(
        transformers=[
            (
                "imputed_num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                dataset_data.get("scale_columns", []),
            ),
            (
                "ordinal_passthrough",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                dataset_data.get("ordinal_columns", []),
            ),
            (
                "non_scaled_passthrough",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                dataset_data.get("non_scaled_columns", []),
            ),
        ],
        remainder="drop",
    )


def make_linear_price_comparison_dataset(df_input, target_column, id_column):
    """Build a numeric dataset for direct price and price-per-square-foot tests."""
    df_work = df_input.copy()
    df_work["PricePerSqFt"] = df_work[target_column] / df_work["GrLivArea"]
    df_work = df_work[np.isfinite(df_work["PricePerSqFt"])].copy()

    # Trim extreme price-per-square-foot outliers before linear/distance models.
    cutoff = df_work["PricePerSqFt"].quantile(0.99)
    df_work = df_work[df_work["PricePerSqFt"] <= cutoff].copy()

    X = (
        df_work.select_dtypes(include="number")
        .drop(columns=[target_column, "PricePerSqFt", id_column], errors="ignore")
        .copy()
    )
    y = df_work[target_column].copy()

    return {
        "X": X,
        "y": y,
        "y_log": log_target(y),
        "y_log_ppsqft": log_target(df_work["PricePerSqFt"].rename("PricePerSqFt")),
        "target": "LogSalePrice and LogPricePerSqFt",
        "cat_features": [],
        "ordinal_columns": get_ordinal_columns(X),
        "scale_columns": get_scale_columns(X),
        "non_scaled_columns": [],
        "requires_scaling": True,
        "model_family": "linear_distance_based",
        "description": (
            "Numeric dataset trimmed at the 99th percentile of PricePerSqFt. "
            "Designed for linear, KNN, and similar price-per-square-foot experiments."
        ),
        "metadata": {
            "price_per_sqft_cutoff": float(cutoff),
        },
    }


def make_processed_full_numeric_dataset(df_input, target_column, id_column):
    """Build the prepared numeric-only dataset."""
    X, y = split_X_y(df_input, target_column=target_column, id_column=id_column)
    X = X.select_dtypes(include="number").copy()

    return {
        "X": X,
        "y": y,
        "y_log": log_target(y),
        "target": "LogSalePrice",
        "cat_features": [],
        "ordinal_columns": get_ordinal_columns(X),
        "scale_columns": get_scale_columns(X),
        "non_scaled_columns": [],
        "requires_scaling": False,
        "model_family": "numeric_models",
        "description": (
            "All numeric columns after common preparation. "
            "Remaining non-numeric categorical columns are excluded."
        ),
    }


def make_processed_full_onehot_dataset(df_input, target_column, id_column):
    """Build a full-feature dataset with remaining categoricals one-hot encoded."""
    X, y = split_X_y(df_input, target_column=target_column, id_column=id_column)
    X = prepare_onehot_features(X)

    return {
        "X": X,
        "y": y,
        "y_log": log_target(y),
        "target": "LogSalePrice",
        "cat_features": [],
        "ordinal_columns": get_ordinal_columns(X),
        "scale_columns": get_scale_columns(X),
        "non_scaled_columns": [],
        "requires_scaling": False,
        "model_family": "numeric_models",
        "encoding": "onehot",
        "description": (
            "All prepared features with remaining categorical columns one-hot encoded. "
            "Designed for sklearn tree-based models that should use the full feature set."
        ),
    }


def make_catboost_dataset(df_input, target_column, id_column):
    """Build a mixed numeric/categorical dataset for CatBoost."""
    X, y = split_X_y(df_input, target_column=target_column, id_column=id_column)
    cat_features = categorical_columns(X)

    for column in cat_features:
        X[column] = X[column].fillna("Missing").astype(str)

    return {
        "X": X,
        "y": y,
        "y_log": log_target(y),
        "target": "LogSalePrice",
        "cat_features": cat_features,
        "requires_scaling": False,
        "model_family": "boosting_categorical",
        "description": (
            "Mixed numeric/categorical prepared dataset for CatBoost. "
            "Ordinal features are numeric and remaining categoricals stay as strings."
        ),
    }


def build_dataset_registry(df_input, target_column, id_column):
    """Create all train dataset variants used by configured candidates."""
    recipes = {
        "linear_price_comparison": make_linear_price_comparison_dataset,
        "processed_full_numeric": make_processed_full_numeric_dataset,
        "processed_full_onehot": make_processed_full_onehot_dataset,
        "catboost_processed_mixed": make_catboost_dataset,
    }
    return {
        name: recipe(df_input, target_column=target_column, id_column=id_column)
        for name, recipe in recipes.items()
    }


def build_dataset_summary(dataset_registry):
    """Summarize dataset variants for run artifacts and review."""
    rows = []
    for name, data in dataset_registry.items():
        rows.append(
            {
                "dataset_version": name,
                "rows": int(data["X"].shape[0]),
                "features": int(data["X"].shape[1]),
                "target": data.get("target"),
                "categorical_features": len(data.get("cat_features", [])),
                "requires_scaling": data.get("requires_scaling", False),
                "model_family": data.get("model_family", ""),
                "description": data.get("description", ""),
            }
        )
    return pd.DataFrame(rows)


def build_test_dataset(dataset_key, train_dataset, prepared_test_df, target_column, id_column):
    """Prepare test features so their columns match the selected train dataset."""
    if train_dataset.get("encoding") == "onehot":
        X_test = prepared_test_df.drop(columns=[target_column, id_column], errors="ignore").copy()
        X_test = prepare_onehot_features(X_test)
        # Test can have missing or unseen dummy columns; align to train schema.
        X_test = X_test.reindex(columns=train_dataset["X"].columns, fill_value=0.0)
    elif train_dataset.get("cat_features"):
        X_test = prepared_test_df.drop(columns=[target_column, id_column], errors="ignore").copy()
        for column in train_dataset.get("cat_features", []):
            if column in X_test.columns:
                X_test[column] = X_test[column].fillna("Missing").astype(str)
        X_test = X_test.reindex(columns=train_dataset["X"].columns)
    else:
        X_test = (
            prepared_test_df.select_dtypes(include="number")
            .drop(columns=[target_column, id_column], errors="ignore")
            .copy()
        )
        X_test = X_test.reindex(columns=train_dataset["X"].columns)

    return {
        "X": X_test,
        "cat_features": train_dataset.get("cat_features", []),
        "target": train_dataset.get("target"),
    }
