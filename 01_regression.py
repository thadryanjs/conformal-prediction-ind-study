# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: title,-widgets,-varInspector
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   title: Conformal Prediction in Regression
# ---

# %% [markdown]
# # Conformal Prediction for Regression
#
# ## Intuition
# The generalization of conformal prediction to the regression makes intuitive with with the following realization: **a residual is a non-conformity score**. The residuals of a model measure very literally how far a prediction was from the expected value. Thus, the machinery of conformal prediction can be readily generalized.
#
# ## Example case
# ### Setup
# We simulate a simple regression case to illustrate this idea:


# %% [code]
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

# It's a little easier to work with dataframes for this
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
# Now that we have our residuals, we can get the quantile estimate as before:

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
# ### Visualization
# We can visualize the results intuitively:

# %% [code]
# Create the DataFrame for plotting, INCLUDING the desired feature
df = pd.DataFrame(
    {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_conf_low": y_conf_low,
        "y_conf_high": y_conf_high,
        # Add the specific feature column from X_test to this DataFrame
        "Feature1": X_test["Feature1"],
    }
)

# Sort the DataFrame by the feature for a smoother plot
df_sorted = df.sort_values(by="Feature1")


# %% [code]
# Plot the data using the pyplot interface
plt.figure(figsize=(12, 7))

# Plot the actual values as scatter points
plt.scatter(
    df_sorted["Feature1"],
    df_sorted["y_test"],
    s=50,
    alpha=0.6,
    label="Actual Value",
    color="blue",
)

# Plot the predicted values as scatter points
plt.scatter(
    df_sorted["Feature1"],
    df_sorted["y_pred"],
    s=50,
    alpha=0.6,
    label="Predicted Value",
    color="orange",
)

# Plot the shaded confidence interval
plt.fill_between(
    df_sorted["Feature1"],
    df_sorted["y_conf_low"],
    df_sorted["y_conf_high"],
    alpha=0.2,
    label="Prediction Interval",
    color="purple",
)

# Set labels and title
plt.xlabel("Feature 1")
plt.ylabel("Target Value")
plt.title("Actual and Predicted Values vs. Feature 1 with Prediction Intervals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Coverage
# We can now verify that we have a 95% coverage rate:

# %% [code]
# check the coverage rate
coverage_rate = np.mean(
    (df_sorted["y_test"] >= df_sorted["y_conf_low"])
    & (df_sorted["y_test"] <= df_sorted["y_conf_high"])
)

print(f"Coverage rate: {coverage_rate:.2f}")
