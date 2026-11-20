# Exercise 3 — Improving the Insurance Risk Model

## 1. Objective

The objective of this exercise is to improve the prediction of insurance pure premium using Tweedie Generalised Linear Models (GLMs), spline transformations, and LightGBM.

Pure premium is defined as the claim amount divided by exposure. Exposure is also used as a sample weight to account for differences in policy duration.

## 2. Data Preparation

The data were loaded using `load_transform()` and split into training and test sets using `create_sample_split()` from Exercise 1.

Observations with non-positive exposure were excluded. The modelling target was `ClaimAmountCut / Exposure`.

The same fixed train/test split was used throughout the experiments.

## 3. Tweedie GLM and Splines

A Tweedie GLM with variance power 1.5 was used as the baseline. Cubic splines were subsequently applied to `BonusMalus` and `VehPower` to allow for nonlinear relationships.

| Model | Train Deviance | Test Deviance |
|---|---:|---:|
| Initial Tweedie GLM | 78.2272 | 76.2497 |
| Standardised GLM | 74.4076 | 73.5172 |
| Spline GLM | 75.6165 | 74.2993 |

The standardised GLM performed better than the spline model on both datasets. The spline specification did not improve predictive performance under the tested settings. This may reflect the choice of spline complexity and regularisation.

## 4. LightGBM and Hyperparameter Tuning

A LightGBM regressor with the Tweedie objective was trained using exposure weights.

To control overfitting, we used restricted tree depth, regularisation and early stopping.

We conducted a grid search over 12 hyperparameter combinations:

- `num_leaves`: 7, 15, 31
- `min_child_samples`: 100, 200
- `reg_lambda`: 5, 10

The training data were further divided into training and validation subsets. Hyperparameters were selected according to validation Tweedie Deviance, rather than test-set performance.

### Best Hyperparameters

| Parameter | Selected value |
|---|---:|
| num_leaves | 15 |
| min_child_samples | 100 |
| reg_lambda | 5.0 |
| Best iteration | 78 |
| Validation Deviance | 72.0423 |

The selected configuration was refitted using the full training set.

## 5. Final Model Comparison

| Model | Train Deviance | Test Deviance |
|---|---:|---:|
| Standardised GLM | 74.4076 | 73.5172 |
| Spline GLM | 75.6165 | 74.2993 |
| Tuned LightGBM | **71.0691** | **72.5625** |

LightGBM achieved the lowest test deviance among the evaluated models.

The difference between LightGBM training and test deviance suggests some generalisation gap. The model used regularisation and early stopping to limit overfitting.

The parameter search selected the original LightGBM configuration. The final test deviance was slightly lower than the earlier LightGBM result of 72.5667, although the difference was negligible.

## 6. Conclusion

LightGBM provided the strongest predictive performance among the models tested. Increasing GLM flexibility through splines did not automatically improve performance.

The exercise illustrates the importance of evaluating models on unseen data, selecting hyperparameters using validation data, and controlling model complexity.

## 7. Implementation Files

- `ex3_model.py` — Initial Tweedie GLM benchmark
- `ex3_splines.py` — Baseline and spline GLM comparison
- `ex3_lightgbm.py` — Initial LightGBM model
- `ex3_tuning.py` — LightGBM hyperparameter tuning
- `ex3_tuning_results.csv` — Grid search results