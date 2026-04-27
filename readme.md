# Housing Prices Notebook Overview

This notebook is a full end-to-end workflow for the Kaggle House Prices dataset. It starts with exploratory analysis, builds a reusable preprocessing layer, creates several model-specific dataset variants, compares multiple models with consistent validation logic, and ends with helper functions for generating submission files on `test.csv`.

## 1. Exploratory Data Analysis

The first part of the notebook loads `train.csv` and `test.csv`, then inspects:

- dataset shape and schema
- duplicate rows
- missing values
- unique counts per column
- descriptive statistics
- the `SalePrice` distribution

It also plots:

- the raw `SalePrice` histogram
- the `log(SalePrice)` histogram
- scatter plots of `SalePrice` vs each numeric feature
- scatter plots of `log(SalePrice)` vs each numeric feature

This section is used to understand skew, outliers, and which features may have predictive signal.

## 2. Shared Preprocessing

The notebook then defines a common preparation step used by all later models.

Main steps:

- fill categorical missing values that actually mean “absence”, such as `NoGarage`, `NoBasement`, `NoFence`
- ordinal-encode ordered categorical variables using mappings from `data_description.txt`
- exclude `Id` from all training datasets and keep it only for final submission files

Helper functions are defined for:

- splitting features and target
- converting `SalePrice` to and from log space
- stratified train/validation splitting for regression by binning `SalePrice`
- stratified K-fold iteration
- metric calculation in both log space and original price space

The notebook reports:

- RMSE on `log(SalePrice)`
- RMSE on `SalePrice`
- MAE
- R²
- average percent error
- min/max absolute error

## 3. Dataset Recipes

Instead of hard-coding one dataset for every model, the notebook builds named dataset variants.

Current recipes:

- `linear_price_comparison`
  - numeric-only
  - trimmed at the 99th percentile of `PricePerSqFt`
  - used for linear-model, KNN, and some price-per-square-foot experiments
- `processed_full_numeric`
  - all prepared numeric features
  - raw string categoricals are excluded
- `processed_full_onehot`
  - all prepared features
  - remaining categorical string columns are one-hot encoded
  - intended for sklearn tree-based models that should use the full feature set
- `knn_numeric`
  - same base as `processed_full_numeric`
  - intended for scaled distance-based models
- `nn_numeric`
  - same base as `processed_full_numeric`
  - intended for neural networks
- `catboost_processed_mixed`
  - keeps remaining categorical string columns for CatBoost

Later in the notebook, additional derived datasets are created:

- `processed_reduced_numeric`
  - reduced to features whose tree importance is greater than `0.1`
- `nn_numeric_reduced`
  - reduced NN dataset built from `nn_numeric` using the same feature-importance filter

## 4. Correlation and Feature Selection

After preprocessing, the notebook:

- computes correlations on prepared numeric features
- plots a heatmap for the top features most correlated with `SalePrice`

It also trains a decision tree and extracts `feature_importances_`, then uses a threshold of `importance > 0.1` to build reduced datasets. Those reduced datasets are reused by later tree, forest, and neural-network experiments.

## 5. Model Evaluation Strategy

Most models are evaluated with the same structure:

1. split the training data into outer train/validation sets
2. run stratified K-fold cross-validation only on the outer-train portion
3. refit on the full outer-train split
4. score on the untouched outer validation split

This means result tables contain both:

- `cv_*` metrics: average cross-validation performance
- `val_*` metrics: holdout validation performance

This is used consistently for the sklearn and CatBoost models. The neural network uses the same outer split and fold structure, but inside each training run it also uses validation loss for learning-rate scheduling and best-weight selection.

## 6. Models in the Notebook

### Linear models

The notebook tests:

- `LinearRegression`
- `Lasso`
- `Ridge`
- `ElasticNet`

They are run in two target modes:

- direct `log(SalePrice)`
- `log(PricePerSqFt)` converted back to `log(SalePrice)` using `GrLivArea`

Numeric features are standardized with `StandardScaler`, ordinal-encoded features are passed through without scaling, and `Id` is excluded from model input entirely.

### KNN

KNN is evaluated on the same `linear_price_comparison` dataset with a grid over:

- `n_neighbors`
- `weights`
- `p` (distance metric)

It also supports both direct log-price prediction and price-per-square-foot prediction.

### Decision Tree

Decision trees are trained on both:

- `processed_full_numeric`
- `processed_full_onehot`

This allows comparison between:

- numeric-only prepared features
- the full prepared feature set with categoricals one-hot encoded

They are tuned over:

- split criterion
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

A second tree run is performed on the reduced feature set.

### Random Forest

Random forests are tested on:

- `processed_full_numeric`
- `processed_full_onehot`
- `processed_reduced_numeric`

The notebook loops over forest hyperparameters such as:

- number of trees
- split criterion
- depth
- leaf constraints
- feature subsampling

### CatBoost

CatBoost is evaluated on:

- `catboost_processed_mixed`
- `processed_full_numeric`
- `processed_reduced_numeric`

This allows comparison between the native mixed categorical dataset and purely numeric variants.

### Neural Network

The notebook contains a simple PyTorch regression network:

- fully connected feedforward architecture
- ReLU activations
- dropout
- Adam optimizer
- `SmoothL1Loss`
- `ReduceLROnPlateau` scheduler using validation loss

The target is `log(SalePrice)`, and predictions are converted back to original price units for metrics. The notebook also plots train/validation loss over epochs and a validation plot of actual vs predicted prices.

## 7. Submission Generation

The final part of the notebook adds reusable submission helpers.

Main ideas:

- rebuild the matching test feature frame from `test_prepared`
- align test columns to the chosen training dataset recipe
- apply the same dataset-specific transformation to test data
  - numeric-only for numeric recipes
  - one-hot encoding plus column alignment for `processed_full_onehot`
  - preserved string categoricals for `catboost_processed_mixed`
- refit the chosen model on full training data
- predict `SalePrice` for `test.csv`
- save a file with:
  - `Id` from `raw_test`
  - predicted `SalePrice`

The main entry point is:

```python
make_submission(model_name, dataset_key, filename=None)
```

Examples in the notebook show how to create files for linear models, trees, forests, CatBoost, and the neural network.

For sklearn trees and forests, there are now two practical submission paths:

- `processed_full_numeric` for numeric-only prepared features
- `processed_full_onehot` for the full prepared feature set including categorical information

## 8. Current Practical Structure

In short, the notebook does this:

1. inspect and visualize the raw training data
2. apply a shared preparation layer
3. create reusable dataset variants
4. compare multiple model families with consistent validation
5. derive reduced feature sets from tree importance
6. train a simple neural network on prepared numeric features
7. export Kaggle-style submission files for selected model/dataset combinations

This makes the notebook both an experimentation workspace and a submission pipeline.

## 9. Kaggle Scores

Submission results sorted by Kaggle score, lower is better:

1. `CatBoost` on `catboost_processed_mixed` — `0.11991`
2. `linear` on `linear_price_comparison` — `0.13970`
3. `random_forest` on `processed_full_onehot` — `0.14330`
4. `random_forest` on `processed_full_numeric` — `0.14348`
5. `knn` on `linear_price_comparison` — `0.16609`
6. `decision_tree` on `processed_full_numeric` — `0.18643`
7. `decision_tree` on `processed_full_onehot` — `0.19189`
8. `neural_network` on `nn_numeric_reduced` — `0.19464`
