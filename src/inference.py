from pathlib import Path

import joblib
import pandas as pd

from src.train import convert_predictions_to_saleprice
from src.utils import save_json


def predict_test_dataset(fitted_bundle, test_dataset):
    """Generate SalePrice predictions for a prepared test dataset."""
    model = fitted_bundle["model"]
    candidate = fitted_bundle["candidate"]
    y_pred_target = model.predict(test_dataset["X"])
    return convert_predictions_to_saleprice(
        y_pred_target,
        X_features=test_dataset["X"],
        target_strategy=candidate["target_strategy"],
    )


def generate_submission(raw_test_df, predictions, id_column, output_path):
    """Write a Kaggle-style submission file with ID and SalePrice columns."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df = pd.DataFrame(
        {
            id_column: raw_test_df[id_column],
            "SalePrice": predictions,
        }
    )
    submission_df.to_csv(output_path, index=False)
    return submission_df


def save_model_artifact(fitted_bundle, candidate, models_dir):
    """Save the fitted model and candidate metadata to the models directory."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = candidate["name"]

    if fitted_bundle["family"] == "catboost":
        artifact_path = models_dir / f"{artifact_stem}.cbm"
        fitted_bundle["model"].save_model(artifact_path)
    else:
        artifact_path = models_dir / f"{artifact_stem}.joblib"
        joblib.dump(fitted_bundle["model"], artifact_path)

    metadata_path = models_dir / f"{artifact_stem}.metadata.json"
    save_json(candidate, metadata_path)
    return artifact_path
