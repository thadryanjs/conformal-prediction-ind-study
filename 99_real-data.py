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

# %%

# %% [code]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

config = {
    "sample_percent": 0.3,
    "random_state": 42,
    "train_prop": 0.7,
    "test_prop": 0.2,
    "cal_prop": 0.1,
}

random_state = config["random_state"]

data_path = "data/PhysioData/INCART 2-lead Arrhythmia Database.csv"

df = pd.read_csv(data_path).drop(columns=["record"])
# Remove all where type is "Q", it's too rare
df = df[df["type"] != "Q"]

if config["sample_percent"] < 1.0:
    df = df.sample(frac=config["sample_percent"], random_state=random_state)

label_encoder = LabelEncoder()
df["type"] = label_encoder.fit_transform(df["type"])

train_prop = config["train_prop"]
test_prop = config["test_prop"]
cal_prop = config["cal_prop"]
assert round(train_prop + test_prop + cal_prop) == 1.0

# Split the data into train, test, and calibration sets
df_train, df_test = train_test_split(
    df,
    train_size=train_prop,
    stratify=df["type"],
    random_state=random_state,
)

df_cal, df_test = train_test_split(
    df_test,
    train_size=cal_prop / (test_prop + cal_prop),
    stratify=df_test["type"],
    random_state=random_state,
)

print(f"Train set size: {df_train.shape[0]}")
print(f"Test set size: {df_test.shape[0]}")
print(f"Calibration set size: {df_cal.shape[0]}")

X_train = df_train.drop(columns=["type"])
y_train = df_train["type"]

# Train the model
model = RandomForestClassifier(random_state=random_state)
model.fit(X_train, y_train)

# Prepare test data
X_test = df_test.drop(columns=["type"])
y_test = df_test["type"]


# %% [code]
X_cal = df_cal.drop(columns=["type"])
y_cal = df_cal["type"]
cal_pred_probs = model.predict_proba(X_cal)

alpha = 0.05
cal_non_conformity_scores = 1 - cal_pred_probs[np.arange(len(cal_pred_probs)), y_cal]
quantile_score = np.quantile(cal_non_conformity_scores, 1 - alpha)


# %% [code]
def printer(verbose, *args, **kwargs):
    """Custom print function that checks verbosity."""
    if verbose:
        print(*args, **kwargs)


def generate_prediction_sets(
    model, X_test, label_encoder, quantile_score, verbose=True
):
    test_preds_probs = model.predict_proba(X_test)
    labels = label_encoder.classes_

    # List to store prediction sets for all samples
    all_prediction_sets = []

    for i in range(len(X_test)):
        printer(verbose, f"\nSample {i}:")
        pred_prob = test_preds_probs[i]
        pred_nc_scores = 1 - pred_prob
        pred_set = []
        cum_score = 0
        sorted_indices = np.argsort(pred_prob)[::-1]

        for j in sorted_indices:
            pj = pred_prob[j]
            printer(
                verbose,
                f"\tProcessing label: {labels[j]}, Probability: {pj:.4f}, Non-conformity score: {pred_nc_scores[j]:.4f}",
            )
            if cum_score + pred_nc_scores[j] <= quantile_score:
                cum_score += pred_nc_scores[j]
                pred_set.append(labels[j])
                printer(
                    verbose,
                    f"\t\tCurrent prediction set: {pred_set}, cum_score: {cum_score:.4f}",
                )
            else:
                printer(
                    verbose,
                    f"\t\tBreaking, cum_score: {cum_score:.4f} + {pred_nc_scores[j]:.4f} > {quantile_score:.4f}",
                )
                break

        printer(verbose, f"\tPredicted set for sample {i}: {pred_set}")

        if pred_set == []:
            # return all the labels in order of their probability
            pred_set = labels[sorted_indices].tolist()

        # Append the prediction set for the current sample to the list
        all_prediction_sets.append(pred_set)

    return all_prediction_sets  # Return the list of prediction sets


prediction_sets = generate_prediction_sets(
    model, X_test, label_encoder, quantile_score, verbose=False
)

# Sample 10531:
#         Processing label: N, Probability: 0.6900, Non-conformity score: 0.3100
#                 Breaking, cum_score: 0.0000 + 0.3100 > 0.0600
#         Predicted set for sample 10531: []


# %% [code]
lens = [len(pred_set) for pred_set in prediction_sets]
print(f"Average size of prediction sets: {np.mean(lens):.2f}")
# get the counts of each length
from collections import Counter

length_counts = Counter(lens)
