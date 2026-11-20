
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, SplineTransformer
)
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_tweedie_deviance

from ps3.data._load_transform import load_transform
from ps3.data._sample_split import create_sample_split

# Load data
df = load_transform()
df = create_sample_split(df, "IDpol", 0.8)
df = df[df["Exposure"] > 0].copy()

train = df[df["sample"] == "train"]
test = df[df["sample"] == "test"]

categorical = ["Area", "VehBrand", "VehGas", "Region"]
nonlinear = ["BonusMalus", "VehPower"]
other_numeric = ["VehAge", "DrivAge", "Density"]
features = categorical + nonlinear + other_numeric

y_train = train["ClaimAmountCut"] / train["Exposure"]
y_test = test["ClaimAmountCut"] / test["Exposure"]

# Compare baseline vs spline using consistent scaling
for name, use_splines in [
    ("Baseline GLM", False),
    ("Spline GLM", True),
]:
    nonlinear_transform = (
        SplineTransformer(n_knots=5, degree=3)
        if use_splines
        else StandardScaler()
    )

    preprocessing = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("nonlinear", nonlinear_transform, nonlinear),
        ("numeric", StandardScaler(), other_numeric),
    ])

    model = Pipeline([
        ("preprocessor", preprocessing),
        ("glm", TweedieRegressor(
            power=1.5,
            alpha=1.0,
            max_iter=1000
        )),
    ])

    model.fit(
        train[features],
        y_train,
        glm__sample_weight=train["Exposure"]
    )

    pred_train = np.maximum(
        model.predict(train[features]), 1e-10
    )
    pred_test = np.maximum(
        model.predict(test[features]), 1e-10
    )

    train_dev = mean_tweedie_deviance(
        y_train, pred_train, power=1.5,
        sample_weight=train["Exposure"]
    )
    test_dev = mean_tweedie_deviance(
        y_test, pred_test, power=1.5,
        sample_weight=test["Exposure"]
    )

    print(f"\n{name}")
    print(f"Train deviance: {train_dev:.4f}")
    print(f"Test deviance:  {test_dev:.4f}")
