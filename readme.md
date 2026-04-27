# House Prices Reproducible Pipeline

This repository contains a reproducible machine-learning pipeline for the Kaggle House Prices competition. The notebook is still available for exploration, but the main result no longer depends on manual notebook execution.

The repository is now organized so that a reviewer can:

1. install dependencies,
2. run one command,
3. reproduce the validation table,
4. save the best model artifact,
5. generate a submission file.

## Repository Layout

```text
project/
├── config.py
├── main.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── inference.py
│   ├── train.py
│   └── utils.py
├── models/
├── outputs/
├── train.csv
└── test.csv
```

## Task Setup

- Task type: regression
- Target column: `SalePrice`
- ID column: `Id`
- Main comparison metric: `val_rmse_log_saleprice`

The default automated pipeline evaluates the main classical candidates from the notebook:

- CatBoost on mixed prepared features
- Ridge on the `PricePerSqFt` formulation
- KNN on the `PricePerSqFt` formulation
- Decision Tree on numeric prepared features
- Decision Tree on one-hot encoded full features
- Random Forest on numeric prepared features
- Random Forest on one-hot encoded full features

The best notebook submission so far is CatBoost on `catboost_processed_mixed`.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note:

- `lightgbm` is included because it was used in notebook experiments.
- On macOS, `lightgbm` may additionally require `libomp`.
- The scripted pipeline does not depend on LightGBM.

## Data Placement

Place the Kaggle competition files in the repository root:

- `train.csv`
- `test.csv`
- `data_description.txt`

The default config already points to these filenames.

## Run the Full Pipeline

The reproducible entry point is:

```bash
python main.py --config config.py
```

This command will:

1. load the config,
2. load train and test data,
3. apply shared preprocessing,
4. build reusable dataset variants,
5. evaluate configured models with outer train/validation split plus inner stratified K-fold CV,
6. save the full results table,
7. select the best candidate,
8. fit the best model on the full training set,
9. save a trained model artifact,
10. generate a submission file for `test.csv`.

There is also a dedicated submission entry point:

```bash
python submission.py --config config.py
```

Optional usage with an explicit configured candidate:

```bash
python submission.py --config config.py --candidate catboost_mixed
python submission.py --config config.py --candidate random_forest_onehot --filename submission_rf_onehot.csv
```

## Outputs

After running `main.py`, the pipeline writes:

- `outputs/model_results.csv`
  - validation and CV metrics for every configured candidate
- `outputs/dataset_summary.csv`
  - summary of the derived dataset variants used by the pipeline
- `outputs/best_result.json`
  - the best candidate row and saved artifact paths
- `outputs/run_summary.json`
  - paths and metadata for the run
- `outputs/submission_best.csv`
  - Kaggle-style submission file with `Id` and predicted `SalePrice`
- `outputs/submission_<candidate>.json`
  - metadata for a dedicated submission run from `submission.py`
- `models/<candidate_name>.joblib` or `models/<candidate_name>.cbm`
  - fitted model artifact for the best candidate

## Configuration

The pipeline is controlled by `config.py`.

You can change:

- data paths
- target and ID columns
- random seed
- validation split size
- number of CV folds
- candidate models
- model hyperparameters
- output filenames

The most important sections in `config.py` are:

- `data`
- `task`
- `cv`
- `outputs`
- `candidates`

Each candidate specifies:

- `name`
- `model_key`
- `dataset_key`
- `target_strategy`
- `preprocessor_kind`
- `params`

The separate [submission.py](/Users/Admin/Documents/VSCode/Housing-Prices/submission.py) script follows the same project logic as `main.py`:

- it loads the same config
- builds the same prepared datasets
- uses the same candidate definitions
- uses the same train-time preprocessing
- writes the submission through the same inference path

## Feature Pipeline

The scripted pipeline reuses the same feature logic that was developed in the notebook:

- fill missing categorical values that mean absence, such as `NoGarage` and `NoBasement`
- ordinal-encode ordered categorical features
- exclude `Id` from model input
- build multiple dataset recipes:
  - `linear_price_comparison`
  - `processed_full_numeric`
  - `processed_full_onehot`
  - `catboost_processed_mixed`

This keeps the notebook experiments and the production pipeline aligned.

## Reproducibility Notes

The repository now avoids notebook-only execution for the main workflow:

- `main.py` is the single scripted entry point
- paths are config-driven rather than absolute local paths
- output directories are created automatically
- the pipeline saves machine-readable CSV and JSON outputs
- random seeds are set explicitly

## Notebook

Exploration remains in:

- `notebooks/EDA.ipynb`

The notebook is intended for:

- EDA
- visualizations
- hypothesis testing
- model experiments
- additional comparisons not yet promoted into the scripted pipeline

The root `housing_prices.ipynb` is kept as the working notebook history, while `notebooks/EDA.ipynb` is the packaged exploratory copy.
