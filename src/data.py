from pathlib import Path

import pandas as pd


def load_competition_data(data_config):
    """Load train data and test data from configured CSV paths."""
    train_path = Path(data_config["train_path"])
    test_path = Path(data_config["test_path"])

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else None
    return train_df, test_df


def validate_competition_data(train_df, test_df, target_column, id_column):
    """Check that required target and ID columns are available."""
    if target_column not in train_df.columns:
        raise KeyError(f"Target column '{target_column}' is missing from train data.")

    if id_column not in train_df.columns:
        raise KeyError(f"ID column '{id_column}' is missing from train data.")

    if test_df is not None and id_column not in test_df.columns:
        raise KeyError(f"ID column '{id_column}' is missing from test data.")
