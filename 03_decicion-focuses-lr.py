# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %% [code]
import numpy as np
import torch
import torch.nn.functional as F

# Check for GPU and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

states = [0, 1]
actions = [0, 1]

# Rewards: R(s, a)
rewards = torch.tensor(
    [[-0.45, 0.50], [-0.10, 0.50]], dtype=torch.float32, device=device
)

# Probs: P(s' | s, a)
probs = torch.tensor(
    [[[0.7, 0.3], [0.2, 0.8]], [[0.99, 0.01], [0.99, 0.01]]],
    dtype=torch.float32,
    device=device,
)

max_iterations = 1000
gamma = 0.5
K = 10
inner_lr = 0.01
meta_lr = 0.01
tau = 0.001
n_meta_iterations = 100


# %% [code]
# checked with AI
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


# %% [code]
def soft_bellman_model_expected(s, a, probs_model, rewards_model, q_table, gamma):
    """
    Model-based soft Bellman expected target:
      B_θ Q(s,a) = r_θ(s,a) + γ * sum_{s'} p_θ(s'|s,a) * logsumexp_a' Q(s',a')
    - probs_model: tensor shape [S, A, S] (probabilities, already softmaxed)
    - rewards_model: tensor shape [S, A]
    - q_table: tensor shape [S, A] (use the target Q w̄ here for stability if desired)
    Returns a scalar tensor (on same device/dtype as q_table).
    """
    r_theta = rewards_model[s, a]
    p_s_prime = probs_model[s, a, :]  # shape [S]
    next_soft_v = torch.logsumexp(q_table, dim=1)  # shape [S]
    expected_future = torch.sum(p_s_prime * next_soft_v)  # scalar
    return r_theta + gamma * expected_future


def soft_bellman_buffer_target(
    s, a, reward_from_buffer, next_state_from_buffer, target_q_table, gamma
):
    """
    Buffer-based soft Bellman single-step target:
      reward + γ * logsumexp_a' Q_target(next_state, a')
    - reward_from_buffer: Python float or torch scalar
    - next_state_from_buffer: int
    - target_q_table: tensor shape [S, A] (should be detached/EMA copy for stability)
    Returns a scalar tensor on the same device/dtype as target_q_table.
    """
    # ensure reward is tensor on same device/dtype
    if not torch.is_tensor(reward_from_buffer):
        reward = torch.as_tensor(
            reward_from_buffer, dtype=target_q_table.dtype, device=target_q_table.device
        )
    else:
        reward = reward_from_buffer.to(
            device=target_q_table.device, dtype=target_q_table.dtype
        )
    next_q_values = target_q_table[next_state_from_buffer, :]  # shape [A]
    log_sum_exp = torch.logsumexp(next_q_values, dim=0)  # scalar
    return reward + gamma * log_sum_exp


# set a seed
torch.manual_seed(8675309)
np.random.seed(8675309)

# We need parameters as torch Tensors with requires_grad=True
q_table = torch.rand(len(states), len(actions), requires_grad=True, device=device)
target_q_table = q_table.clone().detach()

rewards_theta = torch.rand(len(states), len(actions), requires_grad=True, device=device)
probs_alpha = torch.rand(
    len(states), len(actions), len(states), requires_grad=True, device=device
)


# %% [code]
def update_q_step_differentiable(
    q_table, ds, da, probs_alpha, rewards_theta, gamma, inner_lr
):
    """
    One differentiable inner-loop gradient step for Q.
    Args:
      q_table: torch.Tensor [S, A], the current Q (may require_grad=True)
      ds, da: ints for sampled state and action
      probs_alpha: torch.Tensor [S, A, S] logits (will be softmaxed inside)
      rewards_theta: torch.Tensor [S, A]
      gamma: float
      inner_lr: float (step size)
    Returns:
      q_new: torch.Tensor [S, A] (q_table after one gradient descent step), requires_grad may be True
    Notes:
      - Uses autograd.grad with create_graph=True so the step is differentiable for higher-order grads.
      - It computes loss = (q_table[ds,da] - B_theta(q_table)[ds,da])**2 and takes gradient w.r.t. q_table.
    """
    # Convert logits to probabilities
    probs_learned = F.softmax(probs_alpha, dim=-1)  # shape [S, A, S]

    # Compute model-based Bellman target using q_table (you may instead use a target_q if desired)
    r_theta = rewards_theta[ds, da]  # scalar tensor
    p_s_prime = probs_learned[ds, da, :]  # shape [S]
    next_soft_v = torch.logsumexp(q_table, dim=1)  # shape [S]
    expected_future = torch.sum(p_s_prime * next_soft_v)  # scalar
    q_bellman = r_theta + gamma * expected_future

    # Per-sample loss for this (s,a)
    loss = (q_table[ds, da] - q_bellman) ** 2

    # Compute gradient of loss w.r.t. entire q_table (create_graph=True to keep graph)
    grads = torch.autograd.grad(loss, q_table, create_graph=True)[0]  # shape [S, A]

    # Take a gradient descent step (returns new tensor, avoids in-place mutation)
    q_new = q_table - inner_lr * grads

    return q_new


# %% [code]
# EMA update
def update_q_table_ema(q_source, q_target, tau):
    """
    Exponential moving average update: q_target := (1 - tau) * q_target + tau * q_source
    - q_source, q_target: torch tensors of same shape (e.g., Q tables)
    - tau: float in [0, 1]
    This does an in-place update of q_target under torch.no_grad().
    """
    with torch.no_grad():
        q_target.data.copy_((1.0 - tau) * q_target.data + tau * q_source.data)


# %% [code]
def update_theta(
    r_theta_param,
    p_alpha_param,
    probs_true,
    rewards_true,
    states,
    actions,
    q_table,  # IMPORTANT: for accurate Eq.(14) this should be q_inner (differentiable) produced by K-step unroll
    target_q_table,  # target Q used for buffer targets (detached EMA)
    gamma,
    d,  # replay buffer dict
    n_meta_iterations,
    learning_rate,
):
    # NOTE: r_theta_param and p_alpha_param must be leaf tensors with requires_grad=True
    optimizer = torch.optim.Adam([r_theta_param, p_alpha_param], lr=learning_rate)
    optimizer.zero_grad()

    # Step 1: Compute grad_true = d L_true / d q (buffer-based targets using target_q_table)
    total_loss_true = torch.tensor(0.0, device=device)
    buffer_keys = list(d.keys())
    if len(buffer_keys) == 0:
        return

    for i in range(n_meta_iterations):
        idx = np.random.choice(buffer_keys)
        entry = d[idx]
        ds, da, dr, ds_prime = entry["s"], entry["a"], entry["r"], entry["s_prime"]

        # Use buffer-based Bellman target with a detached/EMA target Q
        bell_true = soft_bellman_buffer_target(
            ds,
            da,
            torch.tensor(dr, dtype=torch.float32, device=device),
            ds_prime,
            target_q_table,  # target q used for stability
            gamma,
        )
        qi = q_table[ds, da]
        total_loss_true = total_loss_true + (qi - bell_true.detach()) ** 2

    l_estimate_true = total_loss_true / float(n_meta_iterations)

    # grad_true: we need it as a vector that will be used as grad_outputs in a VJP.
    # Keep create_graph=True if we will compute higher-order autograd that depends on it.
    grad_true = torch.autograd.grad(
        l_estimate_true, q_table, create_graph=True, retain_graph=True
    )[0]

    # Step 2: Compute grad_theta = d L_theta / d q (model-based Bellman, using q_table)
    total_loss_theta = torch.tensor(0.0, device=device)
    probs_learned = F.softmax(p_alpha_param, dim=-1)

    for i in range(n_meta_iterations):
        idx = np.random.choice(buffer_keys)
        entry = d[idx]
        ds, da = entry["s"], entry["a"]

        # Use model-based Bellman expected target; note we pass target_q_table or q_table depending on design
        bell_theta = soft_bellman_model_expected(
            ds, da, probs_learned, r_theta_param, q_table, gamma
        )
        current_q_value_theta = q_table[ds, da]
        total_loss_theta = total_loss_theta + (current_q_value_theta - bell_theta) ** 2

    l_theta = total_loss_theta / float(n_meta_iterations)

    # grad_theta wrt q_table. For accurate IFT, q_table here should be differentiable (q_inner).
    grad_theta = torch.autograd.grad(
        l_theta, q_table, create_graph=True, retain_graph=True
    )[0]

    # Step 3: VJP: map grad_theta -> parameter space contracted by grad_true
    final_grad_products = torch.autograd.grad(
        outputs=grad_theta,
        inputs=[p_alpha_param, r_theta_param],
        grad_outputs=grad_true,
        retain_graph=False,
        allow_unused=True,
    )

    # final_grad_products entries may be None if there's no connection; replace with zeros
    v0 = (
        final_grad_products[0]
        if (final_grad_products is not None and final_grad_products[0] is not None)
        else torch.zeros_like(p_alpha_param)
    )
    v1 = (
        final_grad_products[1]
        if (final_grad_products is not None and final_grad_products[1] is not None)
        else torch.zeros_like(r_theta_param)
    )

    # assign gradients (negative sign from IFT formula)
    p_alpha_param.grad = -v0
    r_theta_param.grad = -v1

    optimizer.step()


# %% [code]
## "Algorithm 1: Model Based RL with OMD Input:"
d = {}
for ir in range(max_iterations):
    if ir == 0:
        s = int(np.random.choice(states))
    else:
        s = s_prime
    action_probs = get_softmax_policies(states, actions, q_table)
    p_values = np.array(list(action_probs[s].values()), dtype=float)
    p_values /= p_values.sum()
    a = int(np.random.choice(actions, p=p_values))
    r = rewards[s, a].item()
    current_trans_probs = probs[s, a, :].cpu().numpy()
    current_trans_probs /= current_trans_probs.sum()
    s_prime = int(np.random.choice(states, p=current_trans_probs))
    d[ir] = {"s": s, "a": a, "r": r, "s_prime": s_prime}

    # 2) fast inner-loop used for acting/training (non-differentiable in-place)
    probs_learned_now = F.softmax(probs_alpha, dim=-1)
    for ik in range(k):
        # sample a transition to update Q (in-place, fast)
        d_index = np.random.choice(list(d.keys()))
        d_entry = d[d_index]
        ds, da = d_entry["s"], d_entry["a"]
        q_bellman = soft_bellman_model_expected(
            ds, da, probs_learned_now, rewards_theta, target_q_table, gamma
        )
        # use the non-differentiable elementwise update for the live q_table
        with torch.no_grad():
            q_table[ds, da] -= inner_lr * 2.0 * (q_table[ds, da] - q_bellman.detach())

    update_q_table_ema(q_table, target_q_table, tau)

    # 4) Accurate meta-update: construct a differentiable q_inner by unrolling K differentiable steps
    #    Start from a detached copy of current q_table, but require grad so autograd records the path
    q_inner = q_table.clone().detach().requires_grad_(True)
    for ik in range(k):
        # sample a transition (can be the same sampling scheme)
        d_index = np.random.choice(list(d.keys()))
        d_entry = d[d_index]
        ds, da = d_entry["s"], d_entry["a"]
        q_inner = update_q_step_differentiable(
            q_inner, ds, da, probs_alpha, rewards_theta, gamma, inner_lr
        )
    # Now q_inner encodes the differentiable K-step optimization path

    # 5) call update_theta which expects a differentiable q_inner for accurate Eq. (14)
    update_theta(
        rewards_theta,
        probs_alpha,
        probs,  # true probs (used only for reference in some variants)
        rewards,
        states,
        actions,
        q_inner,  # pass differentiable q_inner here
        target_q_table,
        gamma,
        d,
        n_meta_iterations,
        learning_rate,
    )


# %% [code]
print("Training finished.\n")

# Final Q and learned model
print("Final Q-table (detached):")
print(q_table.detach().cpu().numpy())

print("\nFinal learned reward parameters (rewards_theta):")
print(rewards_theta.detach().cpu().numpy())

print("\nTrue reward parameters:")
print(rewards.cpu().numpy())

probs_learned = F.softmax(probs_alpha, dim=-1).detach()
print("\nFinal learned transition probabilities (probs softmaxed):")
print(probs_learned.cpu().numpy())

print("\nTrue transition probabilities:")
print(probs.cpu().numpy())

# Greedy policy from final Q
print("\nGreedy policy from final Q-table:")
for s in states:
    greedy_a = int(torch.argmax(q_table[s]).item())
    print(f"  State {s}: take action {greedy_a}")

# Small evaluation using greedy policy on the true environment
def evaluate_policy_greedy(policy_q, env_probs, env_rewards, n_episodes=50, max_steps=50, gamma_eval=0.99):
    total_returns = []
    for ep in range(n_episodes):
        s = int(np.random.choice(states))
        G = 0.0
        discount = 1.0
        for t in range(max_steps):
            a = int(torch.argmax(policy_q[s]).item())
            r = float(env_rewards[s, a].cpu().numpy())
            trans = env_probs[s, a, :].cpu().numpy()
            trans = trans / trans.sum()
            s = int(np.random.choice(states, p=trans))
            G += discount * r
            discount *= gamma_eval
        total_returns.append(G)
    return np.mean(total_returns), np.std(total_returns)

mean_ret, std_ret = evaluate_policy_greedy(q_table.detach(), probs, rewards, n_episodes=100, max_steps=50)
print(f"\nEvaluation of greedy policy on true MDP over 100 episodes: mean return = {mean_ret:.3f}, std = {std_ret:.3f}")

# Diagnostics: MSE between learned and true rewards, and between learned and true transition probs
mse_rewards = torch.mean((rewards_theta.detach() - rewards)**2).item()
mse_probs = torch.mean((probs_learned - probs.detach())**2).item()
print(f"\nDiagnostics:")
print(f"  MSE(rewards): {mse_rewards:.6f}")
print(f"  MSE(transition probs): {mse_probs:.6f}")

# Optional: show per-(s,a) errors
print("\nPer (state,action) errors (rewards, transition L2):")
for s in states:
    for a in actions:
        r_err = (rewards_theta.detach()[s,a] - rewards[s,a]).item()
        p_err = torch.norm(probs_learned[s,a,:] - probs[s,a,:]).item()
        print(f"  (s={s}, a={a}): reward_err = {r_err:.4f}, trans_L2 = {p_err:.4f}")

print("\nDone.")

