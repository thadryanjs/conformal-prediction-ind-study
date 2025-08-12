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
# ## Conformal Prediction in one sentence
#
# Conformal Prediciton allows users to create rigorous estimates of uncertainty similar to confidence intervals regardless of the machine learning model used by applying a non-conformity function to a calibration set of the data an analyzing new data based on where that fall in the quantiles of that non-conformity score.
#
# ## A more reasonable summary
#
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
#
# Essentially, give a new prediction, we want to be able to say "How weird is this?" and select quantiles of acceptable weirdness.
#
# ## An example in Python
# The example below uses the UCI Beans dataset. We will fit a random forest to classify beans by variety. The code is included for the curious. We visualize the dataset:


# %% [code]
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
    "print_plots": True,
}

df = pd.read_excel("data/DryBeanDataset/Dry_Bean_Dataset.xlsx")

df.head()


# %% [markdown]
# We need to set aside 1000 rows of data in order to estimate the distribution of the non-conformity scores from which we will compute $\hat q$. This is the sample size reccomended by the creators basd on emperical studies.


# %% [code]
# make sure we randomize before selecting the calibration set
df = shuffle(df, random_state=config["random_state"])

# Split the data into features and target
X = df.drop(columns=["Class"])  # pyright: ignore
y = df["Class"]  # pyright: ignore


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


# %% [markdown]
# Now that we have a model we can see the predictions is gives as so we can characterize the distribution of the non-conformity scores, ie, the distribution of "weirdness". For our weirdness score we will simply use $1-p(true)$. This is a common non-nonformity score.


# %% [code]
# this is just a bit of housekeeping related to mapping categorical labels to integers
le = LabelEncoder()
y_cal_enc = le.fit_transform(y_cal)

# predictions on the calibration set
cal_preds = rf.predict_proba(X_cal)

# get the predicted probabilities for the true class
# Note: this is an extreme terse numpy idiom, but it just extracts the predicted probabilities for the true class. The arange is just setting up an index of 1 to the lenge of the array
cal_p_true = cal_preds[np.arange(len(y_cal_enc)), y_cal_enc]  # pyright: ignore

# the famed non-conformity score
nc_scores = 1 - cal_p_true

# we want to file the 95% quantile (with a finite sample correction)
qhat = np.quantile(nc_scores, 0.95) * (len(nc_scores) / (len(nc_scores) - 1))

print(f"Quantile estimate (qHat): {qhat:.4f}")


# %% [markdown]
# We have a score based on the quantile of "weirdness" that we can use as a benchmark for future predictions. We can see it in context to make sure we've gotten what we think we have:


# %% [code]
# visualize the distributions of the predicted probabilities
if config["print_plots"]:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(nc_scores, bins=30, kde=True)
    plt.axvline(
        qhat, color="red", linestyle="--", label=f"qHat = {qhat:.2f}"  # pyright: ignore
    )
    plt.title("Distribution of Non-Conformity Scores")
    plt.xlabel("Non-Conformity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# %% [markdown]
# We can return to our modeling. We can make predictions on the test data and build prediction sets from that have $1-\alpha$ coverage guarantees. Our raw probability estimates:


# %% [code]
# Encode the true class labels - just to deal with integers/string names
y_test_enc = le.transform(y_test)

# predict "probabilities" for the test set
test_preds = rf.predict_proba(X_test)

# this gives us the estimate probability of each class being the true class
print(test_preds[:10])


# %% [markdown]
# Now we build the sets by comparing the predicted probabilities to the quantile threshold and taking those that aren't rule out by their "weirdness", thus pushing our sets below the desired quantile. We execute this and instpect:


# %% [code]
# Create the predicted sets based on the quantile threshold
pred_sets = 1 - test_preds <= qhat  # pyright: ignore


print(pred_sets[:25])


# %% [markdown]
# Note that some of these contain more than one predicted class. This is where the theory of conformal prediction maps satisfyingly to intuition, the more uncertainty in the model, the bigger the set we must take to keep the coverage we want. As in the case with sensitivity-specifity tradeoffs, we see the extreme endpoint gives us 100% coverage at the expense of being at all informative: a classifier that predicts "no" for everything never makes a false positive. The analogous phenomenon in conformal prediction is predicting every new observation must be in the set of all possible labels. We can see that, indeed, some of our predictions have more than one possible outcome:


# %% [code]
print(pred_sets[20])


# %% [markdown]
# Let's see if we got the coverage we desire. We will do so by checking if the true class is in the predicted set for each prediction.


# %% [code]
# Check if the predicted sets contain the true class
# Note: This is an extremely terse idiom that optimizes for speed, not clarity.
# It replaces a nested loop checking each class for each sample against the true class
true_class_present = pred_sets[np.arange(len(y_test_enc)), y_test_enc]

print(true_class_present)


# %% [code]
# Calculate the coverage rate
# Note: np.mean True as 1 and False as 0, which is why this works.
coverage_rate = np.mean(true_class_present)

print(f"Manual Coverage rate: {coverage_rate:.2f}")


# %% [markdown]
# We observe that out prediction sets give us a 95% coverage.
#
# ### The MAPIE package
# In real-world uses, a pre-written library may be used. The MAPIE package is the standard CP implementation. We will use it to see if we get similar results.


# %% [code]
rf_mapie = MapieClassifier(estimator=rf, cv="prefit", method="score")

rf_mapie.fit(X_cal, y_cal)

y_pred, y_set = rf_mapie.predict(X_test, alpha=0.05)

mapie_coverage_rate = rf_mapie.score(X_test, y_test)

print(f"MAPIE coverate rate: {mapie_coverage_rate:.2f}")


# %% [markdown]
# There are probably subtle implementation details here that explain the slight difference, but we've show that we have a reasonable demonstration.
#
# # Further Reading
#
# - [This introductory blog post](https://mindfulmodeler.substack.com/p/week-1-getting-started-with-conformal)
# - [This introductory paper](https://arxiv.org/abs/2107.07511)
