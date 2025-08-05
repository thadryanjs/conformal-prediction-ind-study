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


# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

config = {"random_state": 8675309, "test_size": 0.3, "print_plots": False}

df = pd.read_excel("data/DryBeanDataset/Dry_Bean_Dataset.xlsx")

X = df.drop(columns=["Class"])
y = df["Class"]

# temp will be split into calibration and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=config["test_size"], random_state=config["random_state"]
)

# split the remaining data into calibration and test sets
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=1 / 3, random_state=config["random_state"]
)

# fit a random forest model
rf = RandomForestClassifier(random_state=config["random_state"])
rf.fit(X_cal, y_cal)


# %% [code]
le = LabelEncoder()
y_cal_enc = le.fit_transform(y_cal)


# %% [code]
cal_preds = rf.predict_proba(X_cal)
cal_p_true = cal_preds[np.arange(len(y_cal_enc)), y_cal_enc]
nc_scores = 1 - cal_p_true

qhat = np.quantile(nc_scores, 0.95)

print(f"Quantile estimate (qHat): {qhat:.2f}")


# %% [code]
# visualize the distributions of the predicted probabilities
if config["print_plots"]:
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(nc_scores, bins=30, kde=True)
    plt.axvline(qhat, color='red', linestyle='--', label=f'qHat = {qhat:.2f}')
    plt.title("Distribution of Non-Conformity Scores")
    plt.xlabel("Non-Conformity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


pred_sets = (1 - rf.predict_proba(X_test) <= qhat)

print(pred_sets[0:10])  # Print the first 10 prediction sets


for i, pred_set in enumerate(pred_sets[:10]):  # Limit to first 10 samples for brevity
    # map the prediction set back to original class labels
    pred_classes = le.inverse_transform(np.where(pred_set)[0])
    print(f"Sample {i + 1}: Predicted Classes: {pred_classes if len(pred_classes) > 0 else 'None'}")

