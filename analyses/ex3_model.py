
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_tweedie_deviance

from ps3.data._load_transform import load_transform
from ps3.data._sample_split import create_sample_split

# 1. Load data and split
df = load_transform()
df = create_sample_split(df, id_column="IDpol", training_frac=0.8)

# 2. Keep valid exposure
df = df[df["Exposure"] > 0].copy()

train = df[df["sample"] == "train"]
test = df[df["sample"] == "test"]

# 3. Define predictors
categorical = ["Area", "VehBrand", "VehGas", "Region"]
numeric = ["BonusMalus", "VehPower", "VehAge", "DrivAge", "Density"]

features = categorical + numeric

# 4. Preprocess features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]
)

# 5. Build Tweedie GLM
model = Pipeline([
    ("preprocessor", preprocessor),
    ("glm", TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000)),
])

# 6. Define pure premium
y_train = train["ClaimAmountCut"] / train["Exposure"]
y_test = test["ClaimAmountCut"] / test["Exposure"]

# 7. Fit model using exposure weights
model.fit(
    train[features],
    y_train,
    glm__sample_weight=train["Exposure"],
)

# 8. Predict
pred_train = np.maximum(model.predict(train[features]), 1e-10)
pred_test = np.maximum(model.predict(test[features]), 1e-10)

# 9. Calculate Tweedie deviance
train_dev = mean_tweedie_deviance(
    y_train, pred_train, power=1.5,
    sample_weight=train["Exposure"]
)

test_dev = mean_tweedie_deviance(
    y_test, pred_test, power=1.5,
    sample_weight=test["Exposure"]
)

print("Tweedie GLM Benchmark")
print("Train deviance:", train_dev)
print("Test deviance:", test_dev)
