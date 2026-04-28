# Проект по Kaggle House Prices 

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## 0. Как запустить проект

Основной воспроизводимый пайплайн находится не в ноутбуке, а в `main.py`, `config.py` и модулях `src/`.

Установка окружения:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для полного запуска пайплайна:

```bash
python main.py --config config.py
```

Скрипт загружает данные, применяет общий препроцессинг, строит датасеты, обучает и сравнивает модели через кросс-валидацию, выбирает лучшую модель, сохраняет артефакт модели и генерирует submission-файл.

Основные результаты сохраняются в:

- `outputs/model_results.csv`
- `outputs/dataset_summary.csv`
- `outputs/best_result.json`
- `outputs/run_summary.json`
- `outputs/submission_best.csv`
- `models/<candidate_name>.cbm` или `models/<candidate_name>.joblib`

Отдельно можно сгенерировать submission:

```bash
python submission.py --config config.py
python submission.py --config config.py --candidate catboost_mixed
```

`notebooks/EDA.ipynb` используется для EDA, визуализаций и экспериментов. Эксперименты из ноутбука не попадают в `main.py` автоматически: если фича или модель становится финальной, ее нужно перенести в `src/` и `config.py`.


## 1. Exploratory Data Analysis

Проводим обзор `train.csv` датасета. Исследуем количество пустых строк по колонкам
SalePrice имеет смещенное распределение, поэтому применяем логарифмирование для приведения к более близкому к нормальному распределению.

Смотрим какую зависимость имеет SalePrice и log(SalePrice) от признаков, где-то можно проследить линейную зависимость


Смотрим как коррелируют фичи между собой и с SalePrice


## 2. Препроцессинг

Есть 3 типа столбцов: числовые, категориальные и категориальные с иерархией(ordinal). В зависимости от типа производим заполнение пустых значений

 `Id` полностью исключен из обучающих датасетов и используется только при формировании submission-файлов. Для линейных моделей, KNN и нейросети скейлятся только обычные численные признаки, а ordinal-признаки после кодирования остаются без скейлинга. Для деревьев решений, Random Forest и CatBoost отдельный скейлинг не используется.

Для оценки моделей помимо целевой RMSE по log(SalePrice) также смотрим следующие метрики:

- RMSE on `log(SalePrice)`
- RMSE on `SalePrice`
- MAE
- R²
- average percent error
- min/max absolute error


## 3. Описание датасетов

Создаем несколько датасетов для теста разных моделей и фич

- `linear_price_comparison`
  - тестируем предсказание итоговой цены через цену на квадратный фут. Для этого создаем колонку `PricePerSqFt`
  - используем численные признаки
  - делаем срез по 99 перцентилю `PricePerSqFt`
- `processed_full_numeric`
  - используем численные признаки
- `processed_full_onehot`
  - используем все подготовленные признаки
  - оставшиеся категориальные признаки кодируются через one-hot encoding
  - датасет предназначен для sklearn-моделей на деревьях, если хотим учитывать и численные, и категориальные признаки
- `knn_numeric`
  - аналогичен `processed_full_numeric`
  - датасет для KNN модели
- `nn_numeric`
  - аналогичен `processed_full_numeric`
  - датасет для нейросети
- `catboost_processed_mixed`
  - датасет с категориальными фичами для Catboost

Далее будут созданы урезанные по количеству фичей датасеты

- `processed_reduced_numeric`
  - получаем через выбор feature importance > `0.1`
- `nn_numeric_reduced`
  - датасет для нейросети с таким же критерием на основе `nn_numeric` 
- `eda_reduced_numeric`
  - датасет с убранными высокоскореллированными фичами (пары с корреляцией > `0.8`)
- `eda_reduced_nn`
  - версия `eda_reduced_numeric` со скейлингом для нейросети


## 4. Оценка эффективности моделей

Датасеты разбиваются на train и val

Все модели тестируются через кросс валидацию на 5 фолдах, далее происходит обучение на всем train(без val) датасете и считается val score

В таблицу для моделей выводятся:

- `cv_*` - усредненные метрики по кросс валидации
- `val_*`- метрики на валидационном датасете

Для некоторых моделей производится подбор через CV происходит отбор оптимальных гиперпараметров

Здесь опущены некоторые тестируемые значения параметров, показывавшие низкий скор, чтобы уменьшить время компиляции ноутбука

## 5. Тестируемые модели

### Линейные модели

Тестируем следующие:

- `LinearRegression`
- `Lasso`
- `Ridge`
- `ElasticNet`

Для каждой предсказываем:

- напрямую `log(SalePrice)`
- `log(PricePerSqFt)` через `log(SalePrice)` и `GrLivArea`

Скейлим численные фичи через `StandardScaler`, ordinal-encoded фичи оставляем без скейлинга

На `eda_reduced_numeric` модели выдают лучшую целевую метрику `val_rmse_log_saleprice`, однако по остальным метрикам проигрывают стандартному датасету `linear_price_comparison`.

### KNN

Для KNN подбираем следующие параметры:

- `n_neighbors`
- `weights`
- `p` (distance metric)


### Decision Tree

Тестируем на датасетах:

- `processed_full_numeric`
- `processed_full_onehot`
- `eda_reduced_numeric`

Параметры:

- split criterion
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

По feature importance создаем урезанный датасет и тестируем на нем


### Random Forest

Тестируем на датасетах:

- `processed_full_numeric`
- `processed_full_onehot`
- `processed_reduced_numeric`
- `eda_reduced_numeric`

Гиперпараметры:

- number of trees
- split criterion
- depth
- leaf constraints
- feature subsampling

### CatBoost

Гиперпараметры:

- iterations
- learning_rate
- depth
- l2_leaf_reg

Тестируем на датасетах:

- `catboost_processed_mixed`
- `processed_full_numeric`
- `processed_reduced_numeric`
- `eda_reduced_numeric`


### Neural Network

Строим простую нейросеть
Конструктор позволяет добавлять новые полносвязные слои
В качестве функции активации используется ReLu
Также используется dropout, Adam optimizer и LR scheduler
Мы обучаем модель предсказывать логарифм SalePrice, затем конвертируем в целовое значение
В качестве Loss Function изначально была MSELoss, затем SmoothL1Loss - она показала лучшие результаты

Тестировались разные значения dropout, добавление слоев и изменение их размера. Усложнение модели скорее приводит к переобучению и не дает положительных результатов

Строим график для того чтобы понять визуально как отличается предсказанная цена от целевой. При увеличении размера слоев модель давала сильные выбросы на паре значений(порядка 2e6), что ухудшало результат. В целом на урезанном датасете модель ведет себя лучше. Также тестируется `eda_reduced_nn` как версия датасета с удаленными сильно скоррелированными признаками.


## 6. Submission Generation

Для создания файла с предсказанием модели мы обучаем на полном train датасете
В зависимости от модели проводим преобразование train и test датасетов

- для `processed_full_numeric`, `knn_numeric`, `nn_numeric` и производных от них в test оставляются только нужные численные признаки
- для `processed_full_onehot` к `test_prepared` применяется тот же one-hot encoding, после чего колонки выравниваются по train-датасету
- для `catboost_processed_mixed` строковые категориальные признаки сохраняются в исходном виде, как ожидает CatBoost
- в итоговый файл всегда попадают две колонки: `Id` из `test.csv` и предсказанный `SalePrice`


## 7. Результаты

Результаты submission-файлов, отсортированные по Kaggle score. Чем меньше значение, тем лучше:

1. `CatBoost` на `catboost_processed_mixed` — `0.11991`
2. `linear` на `linear_price_comparison` — `0.13970`
3. `random_forest` на `processed_full_onehot` — `0.14330`
4. `random_forest` на `processed_full_numeric` — `0.14348`
5. `knn` на `linear_price_comparison` — `0.16609`
6. `decision_tree` на `processed_full_numeric` — `0.18643`
7. `decision_tree` на `processed_full_onehot` — `0.19189`
8. `neural_network` на `nn_numeric_reduced` — `0.19464`


Лучший результат показал CatBoost

## 8. Идеи и доработки

Можно попытаться улучшить результаты путем работы с фичами. Например, считать возраст дома после ремонта; убрать часть скоррелированных признаков (есть бассейн и его качество; площадь гаража и число машин); сделать менее агрессивный срез по feature importance.
