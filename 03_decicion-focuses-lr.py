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


# [inactive delimiter] [code]
import numpy as np
import torch
import torch.nn.functional as F


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


# [inactive delimiter] [code]
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


# [inactive delimiter] [code]
def update_q_step_differentiable(
    q_table, ds, da, probs_theta, rewards_theta, gamma, inner_lr
):
    """
    One differentiable inner-loop gradient step for Q.
    Args:
      q_table: torch.Tensor [S, A], the current Q (may require_grad=True)
      ds, da: ints for sampled state and action
      probs_theta: torch.Tensor [S, A, S] logits (will be softmaxed inside)
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
    probs_learned = F.softmax(probs_theta, dim=-1)  # shape [S, A, S]

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


# [inactive delimiter] [code]
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


# [inactive delimiter] [code]
def update_theta(
    r_theta_param,
    p_alpha_param,
    probs_true,
    rewards_true,
    states,
    actions,
    q_table,
    target_q_table,
    gamma,
    d,
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


# [inactive delimiter] [code]
def train_omd(
    states,
    actions,
    probs_true,  # torch.tensor shape [S, A, S], true transitions
    rewards_true,  # torch.tensor shape [S, A], true rewards
    probs_theta,
    rewards_theta,
    max_iterations=1000,
    K=10,
    inner_lr=0.01,
    meta_lr=0.01,
    tau=0.001,
    n_meta_iterations=100,
    device=torch.device("cpu"),
    seed=8675309,
):
    """
    Train OMD on a given tabular MDP using Option B (differentiable inner-loop unroll).
    Returns learned q_table, rewards_theta, probs_theta, replay buffer d.
    """

    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    S = len(states)
    A = len(actions)

    # Outer optimizer for theta
    theta_optimizer = torch.optim.Adam([rewards_theta, probs_theta], lr=meta_lr)

    # replay buffer
    d = {}

    # helper locals
    probs_true_np = None
    if isinstance(probs_true, torch.Tensor):
        probs_true_np = probs_true.cpu().numpy()
    else:
        probs_true_np = np.array(probs_true)

    # "Algorithm 1: Model Based RL with OMD Input:"
    for ir in range(max_iterations):
        # sample/advance state
        if ir == 0:
            s = int(np.random.choice(states))
        else:
            s = s_prime

        # get action from softmax policy
        action_probs = get_softmax_policies(states, actions, q_table)
        p_values = np.array(list(action_probs[s].values()), dtype=float)
        p_values /= p_values.sum()
        a = int(np.random.choice(actions, p=p_values))

        # step true MDP: sample next state using probs_true
        trans = probs_true_np[s, a, :].astype(float)
        trans /= trans.sum()
        s_prime = int(np.random.choice(states, p=trans))
        r = float(rewards_true[s, a].item())

        # store transition
        d[ir] = {"s": s, "a": a, "r": r, "s_prime": s_prime}

        # fast in-place inner updates for live q_table
        probs_learned_now = F.softmax(probs_theta, dim=-1)
        for ik in range(K):
            idx = np.random.choice(list(d.keys()))
            entry = d[idx]
            ds, da = entry["s"], entry["a"]
            q_bellman = soft_bellman_model_expected(
                ds, da, probs_learned_now, rewards_theta, target_q_table, gamma=0.5
            )
            with torch.no_grad():
                q_table[ds, da] -= (
                    inner_lr * 2.0 * (q_table[ds, da] - q_bellman.detach())
                )

        # EMA update for target Q
        update_q_table_ema(q_table, target_q_table, tau)

        # Differentiable unroll to build q_inner
        q_inner = q_table.clone().detach().requires_grad_(True)
        for ik in range(K):
            idx = np.random.choice(list(d.keys()))
            entry = d[idx]
            ds, da = entry["s"], entry["a"]
            q_inner = update_q_step_differentiable(
                q_inner,
                ds,
                da,
                probs_theta,
                rewards_theta,
                gamma=0.5,
                inner_lr=inner_lr,
            )

        # accurate meta-update (update_theta must use theta_optimizer or accept optimizer args)
        update_theta(
            rewards_theta,
            probs_theta,
            probs_true,
            rewards_true,
            states,
            actions,
            q_inner,
            target_q_table,
            gamma=0.5,
            d=d,
            n_meta_iterations=n_meta_iterations,
            learning_rate=meta_lr,  # if update_theta creates its own optimizer
        )

        # optionally step a provided optimizer if you pass it in; here update_theta handles stepping.

    return (
        q_table.detach(),
        rewards_theta.detach(),
        F.softmax(probs_theta, dim=-1).detach(),
        d,
    )


torch.manual_seed(8675309)
np.random.seed(8675309)

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
# We need parameters as torch Tensors with requires_grad=True
q_table = torch.rand(len(states), len(actions), requires_grad=True, device=device)
target_q_table = q_table.clone().detach()

rewards_theta = torch.rand(len(states), len(actions), requires_grad=True, device=device)
assert rewards_theta.is_leaf

probs_theta = torch.rand(
    len(states), len(actions), len(states), requires_grad=True, device=device
)
assert probs_theta.is_leaf


max_iterations = 1000
K = 10
inner_lr = 0.01
meta_lr = 0.01
tau = 0.001
n_meta_iterations = 100
seed = 8675309


# use the function
q_table, rewards_theta, probs_theta, d = train_omd(
    states=states,
    actions=actions,
    probs_true=probs,
    rewards_true=rewards,
    probs_theta=probs_theta,
    rewards_theta=rewards_theta,
    max_iterations=max_iterations,
    K=K,
    inner_lr=inner_lr,
    meta_lr=meta_lr,
    tau=tau,
    n_meta_iterations=n_meta_iterations,
    device=device,
    seed=seed,
)

print(q_table)
print(rewards_theta)
print(probs_theta)
