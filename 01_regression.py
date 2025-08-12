# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Conformal Prediction for Regression
#
# # Intuition
# The generalization of conformal prediciton from the classification case to the regression makes becomes intuitive with with the following realization: *a residual is a non-conformity score*. The residuals of a model measure very literally how far a predction was from the expected value. Thus, the machinery of conformal prediction can be readily generalized.
#
# ## Example case
# We simulate a simple regression case to illustrate this idea:


# %% [code]
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("qt5agg")
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.datasets import make_regression

config = {
    "random_state": 8675309,
    "test_size": 0.3,
    "calibration_size": 1000,
}


# Simulate a regression dataset
X, y = make_regression(
    n_samples=10000,
    n_features=5,
    n_informative=5,
    noise=0.5,
    random_state=config["random_state"],
)

# Introduce some non-linear relationships
X[:, 1] = np.sin(X[:, 1])
X[:, 2] = np.cos(X[:, 2])
X[:, 3] = np.sin(X[:, 3])
X[:, 4] = np.cos(X[:, 4])

# It's a litle easer to work with dataframes for this
names = ["Feature" + str(i) for i in range(1, 6)]
df = pd.DataFrame(X, columns=names)
df["Target"] = y

# Shuffle the data and make splits
df = shuffle(df, random_state=config["random_state"])

# Split the data into features and target
X = df.drop(columns=["Target"])  # pyright: ignore
y = df["Target"]  # pyright: ignore

# Select the calibration set
X_cal = X.iloc[: config["calibration_size"]]
y_cal = y.iloc[: config["calibration_size"]]

# Remove the calibration samples
X_remaining = X.iloc[config["calibration_size"] :]
y_remaining = y.iloc[config["calibration_size"] :]

# Make test and train sets
X_train, X_test, y_train, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=config["test_size"],
    random_state=config["random_state"],
)

# Fit a random forest model
rf = RandomForestRegressor(random_state=config["random_state"])
rf.fit(X_train, y_train)

# Compute the residuals
residuals = np.abs(rf.predict(X_test) - y_test)

# %% [markdown]
# # Now that we have our residuals, we can get the quantile estimate as before:

# %% [code]


qhat = np.quantile(residuals, 0.95)

print(f"Quantile estimate (qHat): {qhat:.4f}")

# %% [markdown]
# We will use these residuals to make intervals around our estimates. Intuitively, we can think of this as widening our prediction until we know it covers the desired area. In an almost literal way, we have estimated a boundary and are pushing out our prediction in order to cover it.


# %% [code]
# We make a new prediction on the test set
y_pred = rf.predict(X_test)

# We set a boundary around the prediction using the quantile estimate
y_conf_low = y_pred - qhat
y_conf_high = y_pred + qhat


# %% [markdown]
# We can visualize the results intuitively:

# %% [code]
df = pd.DataFrame(
    {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_conf_low": y_conf_low,
        "y_conf_high": y_conf_high,
    }
)

# Sort the DataFrame by the true values (y_test)
df_sorted = df.sort_values(by="y_test")

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the scatter points
ax.scatter(
    df_sorted["y_test"], df_sorted["y_pred"], s=50, alpha=0.5, label="Predicted Value"
)

# Plot the shaded confidence interval
ax.fill_between(
    df_sorted["y_test"],
    df_sorted["y_conf_low"],
    df_sorted["y_conf_high"],
    alpha=0.2,
    label="Confidence Interval",
)

# Set labels and title
ax.set_xlabel("True Value")
ax.set_ylabel("Predicted Value")
ax.set_title("Prediction with Confidence Interval")
ax.legend()
plt.show()


# %% [code]
# close fig
plt.close()
