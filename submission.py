import argparse
from pathlib import Path

from src.data import load_competition_data, validate_competition_data
from src.features import build_dataset_registry, build_test_dataset, prepare_common_data
from src.inference import generate_submission, predict_test_dataset, save_model_artifact
from src.train import (
    evaluate_candidates,
    fit_full_candidate,
    get_candidate_by_name,
    select_best_candidate,
)
from src.utils import ensure_directories, load_config, save_json, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a House Prices submission file.")
    parser.add_argument(
        "--config",
        default="config.py",
        help="Path to config.py or config.yaml",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Optional candidate name from config.py. If omitted, the script evaluates candidates and uses the best one.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional submission filename. Defaults to the configured output filename or output_<candidate>.csv.",
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
    if test_df is None:
        raise ValueError("Test data is required to generate a submission file.")

    train_prepared = prepare_common_data(train_df)
    test_prepared = prepare_common_data(test_df)

    dataset_registry = build_dataset_registry(
        train_prepared,
        target_column=target_column,
        id_column=id_column,
    )

    results_df = None
    best_result = None
    if args.candidate:
        candidate = get_candidate_by_name(config["candidates"], args.candidate)
    else:
        results_df = evaluate_candidates(
            candidates=config["candidates"],
            dataset_registry=dataset_registry,
            cv_config=config["cv"],
            seed=seed,
            sort_metric=config["pipeline"]["sort_metric"],
        )
        results_path = output_base / config["outputs"]["results_table_filename"]
        results_df.to_csv(results_path, index=False)

        candidate, best_result = select_best_candidate(
            results_df=results_df,
            candidates=config["candidates"],
            sort_metric=config["pipeline"]["sort_metric"],
        )
        print("Selected best candidate from evaluated results:")
        print(best_result.to_dict())

    train_dataset = dataset_registry[candidate["dataset_key"]]
    fitted_bundle = fit_full_candidate(candidate, train_dataset=train_dataset, seed=seed)

    artifact_path = None
    if config["pipeline"]["save_model_artifact"]:
        artifact_path = save_model_artifact(
            fitted_bundle=fitted_bundle,
            candidate=candidate,
            models_dir=models_dir,
        )

    test_dataset = build_test_dataset(
        dataset_key=candidate["dataset_key"],
        train_dataset=train_dataset,
        prepared_test_df=test_prepared,
        target_column=target_column,
        id_column=id_column,
    )
    saleprice_predictions = predict_test_dataset(
        fitted_bundle=fitted_bundle,
        test_dataset=test_dataset,
    )

    if args.filename:
        submission_filename = args.filename
    elif args.candidate:
        submission_filename = f"output_{candidate['name']}.csv"
    else:
        submission_filename = config["outputs"]["submission_filename"]

    submission_path = output_base / submission_filename
    generate_submission(
        raw_test_df=test_df,
        predictions=saleprice_predictions,
        id_column=id_column,
        output_path=submission_path,
    )

    summary_payload = {
        "config_path": args.config,
        "candidate_name": candidate["name"],
        "dataset_key": candidate["dataset_key"],
        "submission_path": str(submission_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
    }
    if best_result is not None:
        summary_payload["selection_metrics"] = best_result.to_dict()

    summary_path = output_base / f"submission_{candidate['name']}.json"
    save_json(summary_payload, summary_path)

    print()
    print(f"Generated submission for candidate: {candidate['name']}")
    print(f"Submission path: {submission_path}")
    if artifact_path:
        print(f"Model artifact path: {artifact_path}")


if __name__ == "__main__":
    main()
