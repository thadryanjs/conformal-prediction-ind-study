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
import torch

states = [0, 1]

actions = [0, 1]

probs = {
    # From state 0
    0: {
        # p(S'=0 | S=0, A=0) and p(S'=1 | S=0, A=0)
        0: {0: 0.7, 1: 0.3},
        # p(S'=0 | S=0, A=1) and p(S'=1 | S=0, A=1)
        1: {0: 0.2, 1: 0.8},
    },
    # From state 1
    1: {
        # p(S'=0 | S=1, A=0) and p(S'=1 | S=1, A=0)
        0: {0: 0.99, 1: 0.01},
        # p(S'=0 | S=1, A=1) and p(S'=1 | S=1, A=1)
        1: {0: 0.99, 1: 0.01},
    },
}

rewards = {
    # From state 0
    0: {
        # r(S=0, A=0)
        0: -0.45,
        # r(S=0, A=1)
        1: 0.50,
    },
    # From state 1
    1: {
        # r(S=1, A=0)
        0: -0.10,
        # r(S=1, A=1)
        1: 0.50,
    },
}

max_iterations = 1000
# Discount factor (immediate rewards vs future rewards)
gamma = 0.5
# number of inner loop iterations (lower is more gradual)
k = 10
# slower learning to prevent issues with numberical stability
learning_rate = 0.01
# EMA factor (also slows down learning)
tau = 0.001


# %(cell) [code]
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


# %(cell) [code]
# πQ(a|s) = expQ(s, a) / sum(expQ(s, a'))
def get_softmax_policies(states, actions, q_table):
    results = {}
    for s in states:
        state_policies = {}
        max_q_value = np.max([q_table[s][a] for a in actions])
        for a in actions:
            # Subtract max to prevent overflow (leading to float errors)
            numerator = np.exp(q_table[s][a] - max_q_value)
            denominator = np.sum([np.exp(q_table[s][a] - max_q_value) for a in actions])
            state_policies[a] = numerator / denominator
        results[s] = state_policies
    return results


# q-table with initial small random values (to test)
q_table = {s: {a: np.random.rand() for a in actions} for s in states}
test_policies = get_softmax_policies(states, actions, q_table)

# print neatly to inspect and confirm they add up to 1
for p in test_policies:
    print(f"State: {p}")
    total = 0
    for a in test_policies[p]:
        print(f"\taction {a}: {test_policies[p][a]}")
        total += test_policies[p][a]
    print(f"\tTotal: {total}")


# %(cell) [code]
# BθQ(s, a) = rθ(s, a) + γEpθ(s′|s,a) log ∑ a′ expQ(s′, a′)
def soft_bellman(s, a, probs, rewards, states, actions, q_table, gamma):
    r_theta = rewards[s][a]
    total = 0
    # for each possible next state...
    for s_prime in states:
        potential_next_s = []
        # ...for each action it could lead to
        for a_prime in actions:  # Iterate over all actions
            # get the value
            potential_next_s.append(q_table[s_prime][a_prime])
        # to the log-sum-exp
        next_soft_v = np.log(np.sum(np.exp(potential_next_s)))
        # account for the probability and add it to the total
        total += probs[s][a][s_prime] * next_soft_v
    return r_theta + gamma * total


# set a seeed
np.random.seed(8675309)

# we need paramters for the inner loop, starting with small random values
# these will be updated in the inner loop

# q-table with initial small random values
q_table = {s: {a: np.random.rand() for a in actions} for s in states}
# the second table is to allow for the EMA proceedure described later
target_q_table = q_table.copy()

# rewards_theta is a randomized version of the rewards (same dimensions)
rewards_theta = {s: {a: np.random.rand() for a in actions} for s in states}

# probs_theta is a randomized version of the probs (same dimensions)
probs_theta = {
    s: {a: {s_prime: np.random.rand() for s_prime in states} for a in actions}
    for s in states
}


# optimization for the inner loop
# minimize: w Es,a[Qw(s, a)−BθQw̄(s, a)]2
def update_q_table(q_table, s, a, q_bellman, learning_rate):
    current_q_value = q_table[s][a]
    error = current_q_value - q_bellman
    q_table[s][a] = q_table[s][a] - learning_rate * (2 * error)

# this is called after the inner loop in to make sure wide swings don't introduce error
def update_q_table_ema(q_table1, q_table2, states, actions, tau):
    for s_key in states:
        for a_key in actions:
            q_table2[s_key][a_key] = (1 - tau) * q_table2[s_key][a_key] + tau * q_table1[s_key][a_key]


# ∂Ltrue(Q*)/∂θ = (∂Ltrue/∂Q*) ⋅ (∂Q*/∂θ)
def update_theta(r_thea, p_theta, probs, rewards, states, actions,
                 q_table, target_q_table, gamma, d, n_interations):

    # We use a dummy optimizer here to manage gradient updates.
    optimizer = torch.optim.Adam([r_thea, p_theta])
    optimizer.zero_grad()

    # =====================================================================
    # Step 1: Compute grad Bellman (d(L_true)/dw)
    # =====================================================================

    total_loss_true = 0.0
    for i in range(0, n_interations):
        # select a random entry from the replay buffer
        d_index = np.random.choice(list(d.keys()))
        d_entry = d[d_index]

        # unpack the entry
        ds, da, dr, ds_prime = d_entry["s"], d_entry["a"], d_entry["r"], d_entry["s_prime"]

        # Calculate true Bellman target using the target_q_table
        bell_true = soft_bellman(ds, da, probs, rewards, states, actions,
                                 target_q_table, gamma, next_state_from_buffer=ds_prime,
                                 reward_from_buffer=dr)

        qi = q_table[ds, da]

        # The L_true loss
        loss = (bell_true - qi) ** 2
        total_loss_true += loss

    l_estimate_true = total_loss_true / n_interations

    grad_true = torch.autograd.grad(l_estimate_true, q_table,
                                   create_graph=True, retain_graph=True)[0]

    # =====================================================================
    # Step 2: Compute grad L_theta (d(L_theta)/dw)
    # =====================================================================

    total_loss_theta = 0.0
    for i in range(0, n_interations):
        # select a random entry from the replay buffer
        d_index = np.random.choice(list(d.keys()))
        d_entry = d[d_index]

        # unpack the entry
        ds, da = d_entry["s"], d_entry["a"]

        current_q_value_theta = q_table[ds, da]

        # Calculate model-induced Bellman target using the learned model and target_q_table
        bell_theta = soft_bellman(ds, da, p_theta, r_theta, states, actions, target_q_table, gamma)

        loss = (current_q_value_theta - bell_theta) ** 2

        total_loss_theta += loss

    l_theta = total_loss_theta / n_interations

    grad_theta = torch.autograd.grad(l_theta, q_table,
                                     create_graph=True, retain_graph=True)[0]

    # =====================================================================
    # Step 3: The Final Update (VJP + optimizer step)
    # =====================================================================

    # Compute the vector-Jacobian product (the 'approx IFT')
    final_grad_products = torch.autograd.grad(
        outputs=grad_theta,
        inputs=[p_theta, r_theta],
        grad_outputs=grad_true,
        retain_graph=False
    )

    # The gradients are returned in the same order as the inputs list.
    grad_p_theta_list = final_grad_products[0:len(p_theta)]
    grad_r_theta_list = final_grad_products[len(p_theta):]

    # Set the gradients for each tensor in the dictionaries
    for grad, param in zip(grad_p_theta_list, p_theta.values()):
        param.grad = -grad

    for grad, param in zip(grad_r_theta_list, r_thea.values()):
        param.grad = -grad

    # Perform the update
    optimizer.step()


# %(cell) [code]
## "Algorithm 1: Model Based RL with OMD  Input:"
## "Initial parameters w, θ, empty replay buffer D."
## "repeat"
d = {}
for ir in range(0, max_iterations):
    ## "Set s to be the current state."
    if ir == 0:
        s = np.random.choice(states)
    else:
        s = s_prime
    ## "Sample an action a using softmax over Qw(s, a)."
    action_probs = get_softmax_policies(states, actions, q_table)
    current_state_policies = action_probs[s]
    a = np.random.choice(actions, p=list(current_state_policies.values()))
    ## "Apply a to get r = r(s, a), s′ ∼ p(s′|s, a)."
    # The reward part
    r = rewards[s][a]
    # the s' part
    current_trans_probs = probs[s][a]
    potential_next_states = list(current_trans_probs.keys())
    s_prime = np.random.choice(
        potential_next_states, p=list(current_trans_probs.values())
    )
    # "Append (s, a, s′, r) to buffer D."
    d[ir] = {"s": s, "a": a, "r": r, "s_prime": s_prime}
    ## for i = 1 to K do
    ## "We make K steps of an optimization method to approximate w∗ = φ(θ) where K is
    ## a hyperparameter and reuse the weights from the previous outer loop iterations."
    for ik in range(1, k):
        ## "Sample (s, a) from buffer D."
        d_index = np.random.choice(list(d.keys()))
        d_entry = d[d_index]
        ds = d_entry["s"]
        da = d_entry["a"]
        ## "Apply θ to get r = rθ(s, a), s′ ∼ pθ(s′|s, a)."
        dr = rewards_theta[ds][da]
        ## "Update Qw parameters w to minimize L(θ, w)."
        ## "BθQ(s, a) = rθ(s, a) + γEpθ(s′|s,a) log ∑ a′ expQ(s′, a′)"
        q_bellman = soft_bellman(
            ds, da, probs_theta, rewards_theta, states, actions, target_q_table, gamma
            )
        update_q_table(q_table, ds, da, q_bellman, learning_rate)

    # back to outer loop
    update_q_table_ema(q_table, target_q_table, states, actions, tau)


    ## "Update model parameters θ according to (14)."
    update_theta(rewards_theta, probs_theta, probs, rewards, states, actions,
                 q_table, target_q_table, gamma, d, n_interations = 100)

