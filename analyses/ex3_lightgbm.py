
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_tweedie_deviance

from ps3.data._load_transform import load_transform
from ps3.data._sample_split import create_sample_split

# 1. Load data
df = load_transform()
df = create_sample_split(df, "IDpol", 0.8)
df = df[df["Exposure"] > 0].copy()

train = df[df["sample"] == "train"].copy()
test = df[df["sample"] == "test"].copy()

categorical = ["Area", "VehBrand", "VehGas", "Region"]
numeric = ["BonusMalus", "VehPower", "VehAge", "DrivAge", "Density"]
features = categorical + numeric

# 2. Prepare categorical features
for col in categorical:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")
    test[col] = test[col].cat.set_categories(train[col].cat.categories)

# 3. Pure premium and exposure weights
X = train[features]
y = train["ClaimAmountCut"] / train["Exposure"]
w = train["Exposure"]

X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
    X, y, w, test_size=0.2, random_state=42
)

# 4. Train LightGBM with regularization
model = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=15,
    max_depth=6,
    min_child_samples=100,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    verbosity=-1,
    random_state=42,
    n_jobs=4,
)

model.fit(
    X_fit, y_fit,
    sample_weight=w_fit,
    eval_set=[(X_val, y_val)],
    eval_sample_weight=[w_val],
    eval_metric="tweedie",
    callbacks=[lgb.early_stopping(30, verbose=False)]
)

# 5. Evaluate
def deviance(X_data, y_true, weights):
    prediction = np.maximum(model.predict(X_data), 1e-10)
    return mean_tweedie_deviance(
        y_true, prediction,
        power=1.5,
        sample_weight=weights
    )

y_test = test["ClaimAmountCut"] / test["Exposure"]

print("\nLightGBM Tweedie")
print("Best iteration:", model.best_iteration_)
print("Train deviance:", round(deviance(X, y, w), 4))
print("Test deviance:", round(deviance(
    test[features], y_test, test["Exposure"]
), 4))