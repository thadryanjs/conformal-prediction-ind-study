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

# %% [markdown]
# # Problem Formulation
# Consider an optimization problem with four variables:
# - `option`: a categorical with five levels: A, B, C, D, E
# - `condition`: a categorical with three levels: condition1, condition2, condition3
# - `satisfaction`: a numeric representing satisfaction on a scale from 1 to 10
# - `cost`: a numerical variable representing cost in monetary units
# In historical data we have all of these variables. When new subjects arrive, we have only condition. We want to predict-then-optimize the satisfaction contrainted by cost. We will estimate the costs from historial data. We will then predict satisfaction for each option and condition, and then optimize the satisfaction under the cost constraint. The twist will be using decision-focused conformal prediction to produce intervals for satisfaction that are cost-aware.

# %% [markdown]
# ```python
# # traditional model predicting cost
# decision_cost_vector = cost_model.coefficients
# # conformal model producing intervals for satisfaction that are cost-aware
# bounds = DFCP(decision_cost_vector)
# # interval linear programming to find the optimal decisions given those bounds
# optimal_decisions = ILP(bounds)
# ```

# %% [code]
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

total_n = 1000
options = ["A", "B", "C", "D", "E"]
conditions = ["condition1", "condition2", "condition3"]

df = pd.DataFrame(
    {
        "option": np.random.choice(options, size=total_n),
        "condition": np.random.choice(conditions, size=total_n),
        "satisfaction": np.random.randint(1, 11, size=total_n),
        "cost": np.random.randint(100, 1000, size=total_n),
    }
)


# make satisfaction a function of option
def satisfaction(satistfaction, option):
    if option == "A":
        res = satistfaction + 1
    elif option == "B":
        res = satistfaction + 2
    elif option == "C":
        res = satistfaction + 3
    elif option == "D":
        res = satistfaction + 4
    elif option == "E":
        res = satistfaction + 5
    else:
        res = np.nan
    if res > 10:
        return 10
    else:
        return res


# make cost a function of option
def cost(cost, option):
    if option == "A":
        return cost + 100
    elif option == "B":
        return cost + 200
    elif option == "C":
        return cost + 300
    elif option == "D":
        return cost + 400
    elif option == "E":
        return cost + 500
    else:
        return np.nan


df["satisfaction"] = df.apply(
    lambda x: satisfaction(x["satisfaction"], x["option"]), axis=1
)

df["cost"] = df.apply(lambda x: cost(x["cost"], x["option"]), axis=1)

df


# %% [code]
# fit a linear model predicting cost from all other variables
X = df.drop(columns=["cost"])
y = df["cost"]

X = pd.get_dummies(X, columns=["option", "condition"], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

# get the decisions and costs from the model
decisions = X_train.columns
costs = model.coef_

# make a dict mapping decisions to costs for future reference
decision_labels_dicts = dict(zip(decisions, costs))
print(decision_labels_dicts)

# make a version where decisions are mapped to integers
decision_costs_int = {i: cost for i, cost in enumerate(costs)}
print(decision_costs_int)


# %% [code]
cost_prod = np.dot(list(decision_costs_int.keys()), list(decision_costs_int.values()))
