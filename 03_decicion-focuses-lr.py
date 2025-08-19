# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# [inactive delimiter] [code]
import numpy as np
import torch
import torch.nn.functional as F

# Check for GPU and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

states = [0, 1]
actions = [0, 1]

# Rewards: R(s, a)
rewards = torch.tensor([
    [-0.45, 0.50],
    [-0.10, 0.50]
], dtype=torch.float32, device=device)

# Probs: P(s' | s, a)
probs = torch.tensor([
    [[0.7, 0.3], [0.2, 0.8]],
    [[0.99, 0.01], [0.99, 0.01]]
], dtype=torch.float32, device=device)


max_iterations = 1000
gamma = 0.5
k = 10
learning_rate = 0.01
tau = 0.001
# rename
n_interations = 100


# [inactive delimiter] [code]
# πQ(a|s) = expQ(s, a) / sum(expQ(s, a'))
def get_softmax_policies(states, actions, q_table):
    """
    Computes softmax policies from the Q-table.
    Explicitly normalizes the probabilities for np.random.choice.
    """
    results = {}
    with torch.no_grad():
        for s in states:
            q_values = q_table[s, :].cpu()
            policy_tensor = F.softmax(q_values, dim=0)

            # Convert to numpy and explicitly normalize to fix potential float errors
            policy_np = policy_tensor.numpy()
            policy_np /= policy_np.sum()

            results[s] = {a: policy_np[i].item() for i, a in enumerate(actions)}
    return results


# [inactive delimiter] [code]
# BθQ(s, a) = rθ(s, a) + γEpθ(s′|s,a) log ∑ a′ expQ(s′, a′)
def soft_bellman(s, a, probs_model, rewards_model, states, actions, q_table, gamma,
                 next_state_from_buffer=None, reward_from_buffer=None):
    """
    Calculates the soft Bellman target using PyTorch tensors.
    """
    if next_state_from_buffer is not None:
        # Use data from the replay buffer
        reward = reward_from_buffer
        next_q_values = q_table[next_state_from_buffer, :]
        log_sum_exp = torch.logsumexp(next_q_values, dim=0)
        return reward + gamma * log_sum_exp
    else:
        # Use the learned model
        r_theta = rewards_model[s, a]

        # Select the transition probabilities and log-sum-exp values
        p_s_prime_sa = probs_model[s, a, :]
        next_soft_v = torch.logsumexp(q_table, dim=1)

        expected_future_value = torch.sum(p_s_prime_sa * next_soft_v)

        return r_theta + gamma * expected_future_value


# set a seed
torch.manual_seed(8675309)
np.random.seed(8675309)

# We need parameters as torch Tensors with requires_grad=True
q_table = torch.rand(len(states), len(actions), requires_grad=True, device=device)
target_q_table = q_table.clone().detach()

# Model parameters to be updated by the outer loop
rewards_theta = torch.rand(len(states), len(actions), requires_grad=True, device=device)
# Probs are parameterized by unconstrained alpha values, which we will learn
probs_alpha = torch.rand(len(states), len(actions), len(states), requires_grad=True, device=device)


# Inner loop update
def update_q_table(q_table, s, a, q_bellman, learning_rate):
    # The PyTorch way: calculate loss and backprop
    current_q_value = q_table[s, a]
    loss = (current_q_value - q_bellman.detach())**2
    # Perform a manual gradient update on the q_table's data
    q_table.data[s, a] -= learning_rate * 2 * (current_q_value - q_bellman.detach())


# EMA update
def update_q_table_ema(q_table1, q_table2, tau):
    with torch.no_grad():
        q_table2.data.copy_((1 - tau) * q_table2.data + tau * q_table1.data)


# The core update function using PyTorch's autograd
def update_theta(r_theta_param, p_alpha_param, probs_true, rewards_true, states, actions,
                 q_table, target_q_table, gamma, d, n_interations, learning_rate):

    # Use a single optimizer to handle all model parameters
    optimizer = torch.optim.Adam([r_theta_param, p_alpha_param], lr=learning_rate)
    optimizer.zero_grad()

    # Step 1: Compute grad Bellman (d(L_true)/dw)
    total_loss_true = torch.tensor(0.0, device=device)
    for i in range(0, n_interations):
        d_index = np.random.choice(list(d.keys()))
        d_entry = d[d_index]
        ds, da, dr, ds_prime = d_entry["s"], d_entry["a"], d_entry["r"], d_entry["s_prime"]

        bell_true = soft_bellman(ds, da, probs_true, rewards_true, states, actions,
                                 target_q_table, gamma, next_state_from_buffer=ds_prime,
                                 reward_from_buffer=torch.tensor(dr, dtype=torch.float32, device=device))

        qi = q_table[ds, da]
        loss = (bell_true.detach() - qi) ** 2
        total_loss_true += loss
    l_estimate_true = total_loss_true / n_interations

    grad_true = torch.autograd.grad(l_estimate_true, q_table, create_graph=True, retain_graph=True)[0]

    # Step 2: Compute grad L_theta (d(L_theta)/dw)
    total_loss_theta = torch.tensor(0.0, device=device)
    probs_learned = F.softmax(p_alpha_param, dim=-1)

    for i in range(0, n_interations):
        d_index = np.random.choice(list(d.keys()))
        d_entry = d[d_index]
        ds, da = d_entry["s"], d_entry["a"]

        current_q_value_theta = q_table[ds, da]
        bell_theta = soft_bellman(ds, da, probs_learned, r_theta_param, states, actions, target_q_table, gamma)
        loss = (current_q_value_theta - bell_theta) ** 2
        total_loss_theta += loss
    l_theta = total_loss_theta / n_interations

    grad_theta = torch.autograd.grad(l_theta, q_table, create_graph=True, retain_graph=True)[0]

    # Step 3: The Final Update (VJP + optimizer step)
    final_grad_products = torch.autograd.grad(
        outputs=grad_theta,
        inputs=[p_alpha_param, r_theta_param],
        grad_outputs=grad_true,
        retain_graph=False
    )

    p_alpha_param.grad = -final_grad_products[0]
    r_theta_param.grad = -final_grad_products[1]

    optimizer.step()


# [inactive delimiter] [code]
## "Algorithm 1: Model Based RL with OMD Input:"
d = {}
for ir in range(0, max_iterations):
    if ir == 0:
        s = np.random.choice(states)
    else:
        s = s_prime

    action_probs = get_softmax_policies(states, actions, q_table)
    current_state_policies = action_probs[s]
    # Explicitly normalize values before np.random.choice
    p_values = np.array(list(current_state_policies.values()))
    p_values /= p_values.sum()
    a = np.random.choice(actions, p=p_values)

    r = rewards[s, a].item()
    current_trans_probs = probs[s, a, :].cpu().numpy()

    # Explicitly normalize for the next state as well, for consistency
    current_trans_probs /= current_trans_probs.sum()

    s_prime = np.random.choice(states, p=current_trans_probs)

    d[ir] = {"s": s, "a": a, "r": r, "s_prime": s_prime}

    for ik in range(1, k):
        d_index = np.random.choice(list(d.keys()))
        d_entry = d[d_index]
        ds, da = d_entry["s"], d_entry["a"]

        probs_learned_inner = F.softmax(probs_alpha, dim=-1)
        q_bellman = soft_bellman(ds, da, probs_learned_inner, rewards_theta, states, actions, target_q_table, gamma)
        update_q_table(q_table, ds, da, q_bellman, learning_rate)

    update_q_table_ema(q_table, target_q_table, tau)

    update_theta(
        rewards_theta, probs_alpha, probs, rewards, states, actions,
        q_table, target_q_table, gamma, d, n_interations, learning_rate
    )



if ir % 100 == 0:
    print("Iteration " + str(ir) + ":")
    print("Q-Table:")
    print(q_table.detach())
    print("\nLearned Reward Parameters:")
    print(rewards_theta.detach())
    print("\nLearned Transition Probabilities:")
    print(F.softmax(probs_alpha, dim=-1).detach())

print("Training finished.")
print("\nFinal Q-Table:")
print(q_table.detach())

print("\nFinal Learned Reward Parameters:")
print(rewards_theta.detach())

print("\nTrue Reward Parameters:")
print(rewards)


print("\nFinal Learned Transition Probabilities:")
print(F.softmax(probs_alpha, dim=-1).detach())

print("\nTrue Transition Probabilities:")
print(probs)

# Print the optimal policy
print("\nOptimal Policy based on Final Q-Table:")
for s in states:
    optimal_action = torch.argmax(q_table[s]).item()
    print(f"In State {s}, the best action is: {optimal_action}")
