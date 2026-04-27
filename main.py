import argparse
from pathlib import Path

from src.data import load_competition_data, validate_competition_data
from src.features import (
    build_dataset_registry,
    build_dataset_summary,
    build_test_dataset,
    prepare_common_data,
)
from src.inference import generate_submission, predict_test_dataset, save_model_artifact
from src.train import evaluate_candidates, fit_full_candidate, select_best_candidate
from src.utils import ensure_directories, load_config, save_json, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run the reproducible House Prices pipeline.")
    parser.add_argument(
        "--config",
        default="config.py",
        help="Path to config.py or config.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config["seed"]
    target_column = config["task"]["target_column"]
    id_column = config["task"]["id_column"]

    output_base = Path(config["outputs"]["base_dir"])
    models_dir = Path(config["outputs"]["models_dir"])
    ensure_directories([output_base, models_dir])
    set_global_seed(seed)

    train_df, test_df = load_competition_data(config["data"])
    validate_competition_data(train_df, test_df, target_column=target_column, id_column=id_column)

    train_prepared = prepare_common_data(train_df)
    test_prepared = prepare_common_data(test_df) if test_df is not None else None

    dataset_registry = build_dataset_registry(
        train_prepared,
        target_column=target_column,
        id_column=id_column,
    )
    dataset_summary = build_dataset_summary(dataset_registry)
    dataset_summary_path = output_base / config["outputs"]["dataset_summary_filename"]
    dataset_summary.to_csv(dataset_summary_path, index=False)

    results_df = evaluate_candidates(
        candidates=config["candidates"],
        dataset_registry=dataset_registry,
        cv_config=config["cv"],
        seed=seed,
        sort_metric=config["pipeline"]["sort_metric"],
    )
    results_path = output_base / config["outputs"]["results_table_filename"]
    results_df.to_csv(results_path, index=False)

    best_candidate, best_result = select_best_candidate(
        results_df=results_df,
        candidates=config["candidates"],
        sort_metric=config["pipeline"]["sort_metric"],
    )

    print("Top candidate results:")
    print(
        results_df[
            [
                "candidate_name",
                "model_key",
                "dataset_key",
                config["pipeline"]["sort_metric"],
                "val_rmse_saleprice",
                "val_mae_saleprice",
                "val_r2_saleprice",
            ]
        ].head(10).to_string(index=False)
    )
    print()
    print("Best candidate:")
    print(best_result.to_dict())

    train_dataset = dataset_registry[best_candidate["dataset_key"]]
    fitted_bundle = fit_full_candidate(best_candidate, train_dataset=train_dataset, seed=seed)

    artifact_path = None
    if config["pipeline"]["save_model_artifact"]:
        artifact_path = save_model_artifact(
            fitted_bundle=fitted_bundle,
            candidate=best_candidate,
            models_dir=models_dir,
        )

    submission_path = None
    if config["pipeline"]["save_submission"] and test_prepared is not None:
        test_dataset = build_test_dataset(
            dataset_key=best_candidate["dataset_key"],
            train_dataset=train_dataset,
            prepared_test_df=test_prepared,
            target_column=target_column,
            id_column=id_column,
        )
        saleprice_predictions = predict_test_dataset(
            fitted_bundle=fitted_bundle,
            test_dataset=test_dataset,
        )
        submission_path = output_base / config["outputs"]["submission_filename"]
        generate_submission(
            raw_test_df=test_df,
            predictions=saleprice_predictions,
            id_column=id_column,
            output_path=submission_path,
        )

    best_result_payload = best_result.to_dict()
    best_result_payload["artifact_path"] = str(artifact_path) if artifact_path else None
    best_result_payload["submission_path"] = str(submission_path) if submission_path else None
    best_result_path = output_base / config["outputs"]["best_result_filename"]
    save_json(best_result_payload, best_result_path)

    run_summary = {
        "config_path": args.config,
        "results_table_path": str(results_path),
        "dataset_summary_path": str(dataset_summary_path),
        "best_result_path": str(best_result_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "submission_path": str(submission_path) if submission_path else None,
    }
    run_summary_path = output_base / config["outputs"]["run_summary_filename"]
    save_json(run_summary, run_summary_path)

    print()
    print(f"Saved results table to {results_path}")
    print(f"Saved dataset summary to {dataset_summary_path}")
    if artifact_path:
        print(f"Saved model artifact to {artifact_path}")
    if submission_path:
        print(f"Saved submission to {submission_path}")


if __name__ == "__main__":
    main()
