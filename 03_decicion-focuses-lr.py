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
def soft_bellman(transition_probs, rewards, states, actions, gamma):
    values = {state: 0 for state in states}
    for s in states:
        action_values = {a: 0 for a in actions}
        for a in actions:
            for next_s in transition_probs[s][a]:
                action_values[a] += transition_probs[s][a][next_s] * (
                    rewards[s][a] + gamma * values[next_s]
                )
        values[s] = np.log(np.sum(np.exp(list(action_values.values()))))
    return values

values = soft_bellman(transition_probs, rewards, states, actions, gamma)


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


# %% [code]
# πQ(a|s) = expQ(s, a) / sum(expQ(s, a'))
def get_softmax_policies(state, actions, q_table):
    results = {}
    for s in states:
        state_policies = {}
        for a in actions:
            numerator = np.exp(q_table[s][a])
            denominator = np.sum([np.exp(q_table[s][a]) for a in actions])
            state_policies[a] = numerator / denominator
        results[s] = state_policies
    return results

test_policies = get_softmax_policies(states, actions, q_table)

# print neatly to inspect and confirm they add up to 1
for p in test_policies:
    print(f"State: {p}")
    total = 0
    for a in test_policies[p]:
        print(f"\taction {a}: {test_policies[p][a]}")
        total += test_policies[p][a]
    print(f"\tTotal: {total}")



# %% [code]
# set a seeed
np.random.seed(8675309)
# q-table with initial small random values
q_table = {s: {a: np.random.rand() for a in actions} for s in states}

# "Algorithm 1: Model Based RL with OMD  Input:"
# "Initial parameters w, θ, empty replay buffer D."
# "repeat"
d = {}
for ir in range(0, max_iterations):
    # "Set s to be the current state."
    if ir == 0:
        s = np.random.choice(states)
    else:
        s = s_prime
    # "Sample an action a using softmax over Qw(s, a)."
    action_probs = get_softmax_policies(s, actions, q_table)
    current_state_policies = action_probs[s]
    a = np.random.choice(actions, p=list(current_state_policies.values()))
    # "Apply a to get r = r(s, a), s′ ∼ p(s′|s, a)."
    # The reward part
    r = rewards[s][a]
    # the s' part
    current_trans_probs = transition_probs[s][a]
    potential_next_states = list(current_transition_probs.keys())
    s_prime = np.random.choice(potential_next_states, p=list(current_trans_probs.values()))
    # Append (s, a, s′, r) to buffer D.
    d[ir] = {"s": s, "a": a, "r": r, "s_prime": s_prime}
    # for i = 1 to K do
    # "We make K steps of an optimization method to approximate w∗ = φ(θ) where K is
    # a hyperparameter and reuse the weights from the previous outer loop iterations."
    for ik in range(1, k):
        # Sample (s, a) from buffer D.
        # Apply θ to get r = rθ(s, a), s′ ∼ pθ(s′|s, a).
        q = soft_bellman(transition_probs, rewards, states, actions, gamma)
        # Update Qw parameters w to minimize L(θ, w).
        pass

