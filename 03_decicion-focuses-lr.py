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

states = [0, 1]

actions = [0, 1]

values = {0: 0, 1: 0}

transition_probs = {
        # From State 0
        0: {
             # p(S'=0 | S=0, A=0) and p(S'=1 | S=0, A=0)
            0: {0: 0.7, 1: 0.3},
            # p(S'=0 | S=0, A=1) and p(S'=1 | S=0, A=1)
            1: {0: 0.2, 1: 0.8}
        },
        # From State 1
        1: {
            # p(S'=0 | S=1, A=0) and p(S'=1 | S=1, A=0)
            0: {0: 0.99, 1: 0.01},
            # p(S'=0 | S=1, A=1) and p(S'=1 | S=1, A=1)
            1: {0: 0.99, 1: 0.01}
        }
    }

rewards = {
        # From State 0
        0: {
            # r(S=0, A=0)
            0: -0.45,
            # r(S=0, A=1)
            1: 0.50
        },
        # From State 1
        1: {
            # r(S=1, A=0)
            0: -0.10,
            # r(S=1, A=1)
            1: 0.50
        }
    }


# Discount factor
gamma = 0.9
max_iterations = 1000
k = 100


# %% [code]
def soft_bellman_operator(transition_probs, rewards, states, actions, gamma):
    values = {state: 0 for state in states}

    for state in states:
        action_values = {action: 0 for action in actions}

        for action in actions:
            for next_state in transition_probs[state][action]:
                action_values[action] += transition_probs[state][action][next_state] * (
                    rewards[state][action] + gamma * values[next_state]
                )

        values[state] = np.log(np.sum(np.exp(list(action_values.values()))))

    return values

values = soft_bellman_operator(transition_probs, rewards, states, actions, gamma)


# %% [code]
# the actual algo
"""
NotebookLM
1. Bi-level Optimization: The Optimal Model Design (OMD) algorithm operates as a bi-level optimization problem.
    ◦ The outer loop is responsible for updating the model parameters θ.
    ◦ The inner loop is responsible for updating the Q-network parameters w for a fixed model θ.

Algorithm 1: Model Based RL with OMD  Input:
Initial parameters w, θ, empty replay buffer D.
repeat
    Set s to be the current state.
    Sample an action a using softmax over Qw(s, a).
    Apply a to get r = r(s, a), s′ ∼ p(s′|s, a).
    Append (s, a, s′, r) to buffer D.
    for i = 1 to K do
        Sample (s, a) from buffer D.
        Apply θ to get r = rθ(s, a), s′ ∼ pθ(s′|s, a).
        Update Qw parameters w to minimize L(θ, w).
    end for
    Update model parameters θ according to (14).
    until the maximum number of interactions is reached
"""

# Algorithm 1: Model Based RL with OMD  Input:
# Initial parameters w, θ, empty replay buffer D.
# repeat
for ir in range(1, max_iterations):
    # Set s to be the current state.
    # Sample an action a using softmax over Qw(s, a).
    # Apply a to get r = r(s, a), s′ ∼ p(s′|s, a).
    # Append (s, a, s′, r) to buffer D.
    # params = ()
    pass
    # for i = 1 to K do
    # "We make K steps of an optimization method to approximate w∗ = φ(θ) where K is
    # a hyperparameter and reuse the weights from the previous outer loop iterations."
    for ik in range(1, k):
        # Sample (s, a) from buffer D.
        # Apply θ to get r = rθ(s, a), s′ ∼ pθ(s′|s, a).
        # Update Qw parameters w to minimize L(θ, w).
        pass


