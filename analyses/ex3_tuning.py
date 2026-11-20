
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_tweedie_deviance

from ps3.data._load_transform import load_transform
from ps3.data._sample_split import create_sample_split


# 1. Load and split data
df = load_transform()
df = create_sample_split(df, "IDpol", 0.8)
df = df[df["Exposure"] > 0].copy()

train = df[df["sample"] == "train"].copy()
test = df[df["sample"] == "test"].copy()

categorical = ["Area", "VehBrand", "VehGas", "Region"]
numeric = [
    "BonusMalus", "VehPower",
    "VehAge", "DrivAge", "Density"
]
features = categorical + numeric

# Make category encoding consistent
for col in categorical:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")
    test[col] = test[col].cat.set_categories(
        train[col].cat.categories
    )

X = train[features]
y = train["ClaimAmountCut"] / train["Exposure"]
w = train["Exposure"]

# Validation comes only from training data
X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
    X, y, w,
    test_size=0.2,
    random_state=42
)


# 2. Evaluation function
def calculate_deviance(model, X_data, y_true, weights):
    predictions = np.maximum(
        model.predict(X_data), 1e-10
    )
    return mean_tweedie_deviance(
        y_true,
        predictions,
        power=1.5,
        sample_weight=weights
    )


# 3. Hyperparameter combinations
param_grid = {
    "num_leaves": [7, 15, 31],
    "min_child_samples": [100, 200],
    "reg_lambda": [5.0, 10.0],
}

combinations = list(itertools.product(
    param_grid["num_leaves"],
    param_grid["min_child_samples"],
    param_grid["reg_lambda"]
))

results = []

print(f"\nTesting {len(combinations)} combinations...\n",
      flush=True)


# 4. Train and evaluate each combination
for i, (leaves, min_samples, reg) in enumerate(
    combinations, start=1
):
    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=leaves,
        max_depth=6,
        min_child_samples=min_samples,
        colsample_bytree=0.8,
        reg_lambda=reg,
        verbosity=-1,
        random_state=42,
        n_jobs=4,
    )

    model.fit(
        X_fit, y_fit,
        sample_weight=w_fit,
        eval_X=X_val,
        eval_y=y_val,
        eval_sample_weight=[w_val],
        eval_metric="tweedie",
        callbacks=[
            lgb.early_stopping(30, verbose=False)
        ],
    )

    train_dev = calculate_deviance(
        model, X_fit, y_fit, w_fit
    )
    val_dev = calculate_deviance(
        model, X_val, y_val, w_val
    )

    results.append({
        "num_leaves": leaves,
        "min_child_samples": min_samples,
        "reg_lambda": reg,
        "best_iteration": model.best_iteration_,
        "train_deviance": train_dev,
        "val_deviance": val_dev,
    })

    print(
        f"[{i}/12] leaves={leaves}, "
        f"min_samples={min_samples}, lambda={reg} | "
        f"Train={train_dev:.4f}, "
        f"Val={val_dev:.4f}",
        flush=True
    )


# 5. Compare all results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "val_deviance"
).reset_index(drop=True)

print("\n===== TUNING RESULTS =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    "analyses/ex3_tuning_results.csv",
    index=False
)

best = results_df.iloc[0]

print("\n===== BEST PARAMETERS =====")
print("num_leaves:", int(best["num_leaves"]))
print("min_child_samples:", int(best["min_child_samples"]))
print("reg_lambda:", best["reg_lambda"])
print("Best iteration:", int(best["best_iteration"]))
print("Validation deviance:", round(best["val_deviance"], 4))


# 6. Refit on all training data
final_model = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    n_estimators=max(1, int(best["best_iteration"])),
    learning_rate=0.05,
    num_leaves=int(best["num_leaves"]),
    max_depth=6,
    min_child_samples=int(best["min_child_samples"]),
    colsample_bytree=0.8,
    reg_lambda=float(best["reg_lambda"]),
    verbosity=-1,
    random_state=42,
    n_jobs=4,
)

final_model.fit(
    X, y,
    sample_weight=w
)

# 7. Final test evaluation (only once)
y_test = test["ClaimAmountCut"] / test["Exposure"]
w_test = test["Exposure"]

final_train_dev = calculate_deviance(
    final_model, X, y, w
)
final_test_dev = calculate_deviance(
    final_model, test[features], y_test, w_test
)

print("\n===== FINAL LIGHTGBM =====")
print("Train deviance:", round(final_train_dev, 4))
print("Test deviance:", round(final_test_dev, 4))
print("Previous LightGBM test deviance: 72.5667")