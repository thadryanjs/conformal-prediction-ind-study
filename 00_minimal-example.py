# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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


# %(cell) [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

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


# %(cell) [code]
le = LabelEncoder()
y_cal_enc = le.fit_transform(y_cal)

cal_preds = rf.predict_proba(X_cal)
cal_p_true = cal_preds[np.arange(len(y_cal_enc)), y_cal_enc]

nc_scores = 1 - cal_p_true

qhat = np.quantile(nc_scores, 0.95) * (len(nc_scores) / (len(nc_scores) - 1))

print(f"Quantile estimate (qHat): {qhat:.4f}")


# %(cell) [code]
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


# %(cell) [code]
# Encode the true class labels for the test set
y_test_enc = le.transform(y_test)  # Use the same encoder to transform y_test

# Predict probabilities for the test set
test_preds = rf.predict_proba(X_test)

# Create the predicted sets based on the quantile threshold
pred_sets = 1 - test_preds <= qhat

# Check if the predicted sets contain the true class
true_class_present = pred_sets[np.arange(len(y_test_enc)), y_test_enc]

# Calculate the coverage rate
coverage_rate = np.mean(true_class_present)

print(f"Manual Coverage rate: {coverage_rate:.2f}")


# %(cell) [code]
rf_mapie = MapieClassifier(estimator=rf, cv="prefit",  method="score")

rf_mapie.fit(X_cal, y_cal)

y_pred, y_set = cp.predict(X_test, alpha=0.05)
mapie_coverage_rate = rf_mapie.score(X_test, y_test)

print(f"MAPIE coverate rate: {mapie_coverage_rate:.2f}")
