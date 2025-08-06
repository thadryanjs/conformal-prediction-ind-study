# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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
#   title: Conformal Prediction for people in a hurry
# ---

# %% [code]

# %% [markdown]
# - **Conformal Prediction (CP)** is a method for getting calibrated uncertainty estimates for machine learning models where they are not generally available, something like confidence intervals for almost any model.
# - Surprisingly, it works for classification problems as well as regression.
#   - The regression case produces a prediction interval, similar to a confidence interval.
#   - The classification case produces sets of predictions that are guaranteed to contain the true class label with a certain probability.
#
# - CP works by estimating a "non-conformity" score for each sample and comparing that to a distribution of non-conformity (NC) scores derived from a subset of the data.
#   - The data are split into test, train, and calibration sets.
#   - NC scores are calculated on the calibration set, and the 1-alpha quantile of these scores (qHat) is noted.
#   - When new data is encountered, the NC of the new data is compared to predicted probabilities for each class.
#   - Potential outcomes are included in the final prediction set if the NC score is less than or equal to qHat.
#   - In the regression case, qHat is derived from the residuals and used to create prediction intervals by adding and subtracting qHat from the predicted value.


# %% [code]
print("no")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from mapie.classification import MapieClassifier

config = {
    "random_state": 8675309,
    "test_size": 0.3,
    "calibration_size": 1000,
    "print_plots": False,
}

df = pd.read_excel("data/DryBeanDataset/Dry_Bean_Dataset.xlsx")

df = shuffle(df, random_state=config["random_state"])

# Split the data into features and target
X = df.drop(columns=["Class"])
y = df["Class"]


X_cal = X.iloc[: config["calibration_size"]]
y_cal = y.iloc[: config["calibration_size"]]

# remove the calibration samples
X_remaining = X.iloc[config["calibration_size"] :]
y_remaining = y.iloc[config["calibration_size"] :]

# make test and train sets
X_train, X_test, y_train, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=config["test_size"],
    random_state=config["random_state"],
)

# fit a random forest model
rf = RandomForestClassifier(random_state=config["random_state"])
rf.fit(X_train, y_train)


# %% [code]
le = LabelEncoder()
y_cal_enc = le.fit_transform(y_cal)

cal_preds = rf.predict_proba(X_cal)
cal_p_true = cal_preds[np.arange(len(y_cal_enc)), y_cal_enc]

nc_scores = 1 - cal_p_true

qhat = np.quantile(nc_scores, 0.95) * (len(nc_scores) / (len(nc_scores) - 1))

print(f"Quantile estimate (qHat): {qhat:.4f}")


# %% [code]
# visualize the distributions of the predicted probabilities
if config["print_plots"]:
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(nc_scores, bins=30, kde=True)
    plt.axvline(qhat, color="red", linestyle="--", label=f"qHat = {qhat:.2f}")
    plt.title("Distribution of Non-Conformity Scores")
    plt.xlabel("Non-Conformity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# %% [code]
# Encode the true class labels - just to deal with integers/string names
y_test_enc = le.transform(y_test)

# predict "probabilities" for the test set
test_preds = rf.predict_proba(X_test)


# %% [code]
# this gives us the estimate probability of each class being the true class
print(test_preds[:10])


# %% [code]
# Create the predicted sets based on the quantile threshold
pred_sets = 1 - test_preds <= qhat

print(pred_sets[:25])


# %% [code]
print(pred_sets[20])


# %% [code]
# Check if the predicted sets contain the true class
# Note: This is an extremely terse idiom that optimizes for speed, not clarity.
# It replaces a nested loop checking each class for each sample against the true class
true_class_present = pred_sets[np.arange(len(y_test_enc)), y_test_enc]

true_class_present


# %% [code]
# Calculate the coverage rate
# Note: np.mean True as 1 and False as 0, which is why this works.
coverage_rate = np.mean(true_class_present)

print(f"Manual Coverage rate: {coverage_rate:.2f}")


# %% [code]
rf_mapie = MapieClassifier(estimator=rf, cv="prefit", method="score")

rf_mapie.fit(X_cal, y_cal)

y_pred, y_set = rf_mapie.predict(X_test, alpha=0.05)
mapie_coverage_rate = rf_mapie.score(X_test, y_test)

print(f"MAPIE coverate rate: {mapie_coverage_rate:.2f}")
