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



# %% [code]
import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Define the number of samples
num_samples = 100

# Create a DataFrame
data = {
    'state': [f'state_{i}' for i in range(num_samples)],
    'action': [f'action_{np.random.randint(1, 5)}' for _ in range(num_samples)],
    'reward': np.random.uniform(0, 10, num_samples),  # Random rewards between 0 and 10
    'next_state': [f'state_{np.random.randint(0, num_samples)}' for _ in range(num_samples)],
    'theta_param': np.random.uniform(-1, 1, num_samples)  # Random parameters for dynamics model
}

df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())



# %% [code]
import numpy as np

def softmax(theta):
    """Compute the softmax probabilities for the given parameters theta."""
    exp_theta = np.exp(theta - np.max(theta))
    return exp_theta / np.sum(exp_theta)

def action_probabilities(state, theta):
    """Compute the action probabilities for a given state using the softmax policy."""
    return softmax(theta)

theta_example = np.array([0.2, 2.0, 0.5])

state = "state_1"
probabilities = action_probabilities(state, theta_example)

print("Action Probabilities for state '{}':".format(state))
for action, prob in enumerate(probabilities):
    print(f"Action {action}: Probability = {prob:.4f}")
