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

# omd_gymnasium_runner.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

# --------------------------
# Simple replay buffer
# --------------------------
class SimpleReplay:
    def __init__(self, capacity=10000):
        self.buf = []
        self.capacity = capacity
    def add(self, s, a, r, ns, done):
        if len(self.buf) >= self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, ns, done))
    def sample(self, n=1):
        assert len(self.buf) > 0, "Replay empty"
        idx = np.random.choice(len(self.buf), size=n)
        return [self.buf[i] for i in idx]
    def __len__(self):
        return len(self.buf)

# --------------------------
# Core helpers (minimal)
# --------------------------
def soft_bellman_model_expected(s, a, probs_model, rewards_model, q_table, gamma):
    r_theta = rewards_model[s, a]
    p_s_prime = probs_model[s, a, :]
    next_soft_v = torch.logsumexp(q_table, dim=1)
    expected_future = torch.sum(p_s_prime * next_soft_v)
    return r_theta + gamma * expected_future

def soft_bellman_buffer_target(ds, da, reward_from_buffer, next_state_from_buffer, done_flag, target_q_table, gamma):
    if not torch.is_tensor(reward_from_buffer):
        reward = torch.as_tensor(reward_from_buffer, dtype=target_q_table.dtype, device=target_q_table.device)
    else:
        reward = reward_from_buffer.to(device=target_q_table.device, dtype=target_q_table.dtype)
    if done_flag:
        return reward
    next_q_values = target_q_table[next_state_from_buffer, :]
    return reward + gamma * torch.logsumexp(next_q_values, dim=0)

def update_q_step_differentiable(q_table, ds, da, probs_theta, rewards_theta, gamma, inner_lr):
    probs_learned = F.softmax(probs_theta, dim=-1)
    r_theta = rewards_theta[ds, da]
    p_s_prime = probs_learned[ds, da, :]
    next_soft_v = torch.logsumexp(q_table, dim=1)
    expected_future = torch.sum(p_s_prime * next_soft_v)
    q_bellman = r_theta + gamma * expected_future
    loss = (q_table[ds, da] - q_bellman) ** 2
    grads = torch.autograd.grad(loss, q_table, create_graph=True)[0]
    q_new = q_table - inner_lr * grads
    return q_new

def update_q_table_ema(q_source, q_target, tau):
    with torch.no_grad():
        q_target.data.copy_((1.0 - tau) * q_target.data + tau * q_source.data)

def update_theta_vjp(
    r_theta_param,
    p_alpha_param,
    states,
    actions,
    q_table,
    target_q_table,
    gamma,
    replay,  # SimpleReplay instance
    n_meta_iterations,
    learning_rate,
    device,
):
    if len(replay) == 0:
        return
    optimizer = torch.optim.Adam([r_theta_param, p_alpha_param], lr=learning_rate)
    optimizer.zero_grad()

    # grad_true from buffer-based targets
    total_loss_true = torch.tensor(0.0, device=device)
    for i in range(n_meta_iterations):
        s, a, r, ns, done = replay.sample(1)[0]
        bell_true = soft_bellman_buffer_target(s, a, r, ns, done, target_q_table, gamma)
        qi = q_table[s, a]
        total_loss_true = total_loss_true + (qi - bell_true.detach()) ** 2
    l_estimate_true = total_loss_true / float(n_meta_iterations)
    grad_true = torch.autograd.grad(l_estimate_true, q_table, create_graph=True, retain_graph=True)[0]

    # grad_theta from model-based targets
    total_loss_theta = torch.tensor(0.0, device=device)
    probs_learned = F.softmax(p_alpha_param, dim=-1)
    for i in range(n_meta_iterations):
        s, a, r, ns, done = replay.sample(1)[0]
        if done:
            bell_theta = r_theta_param[s, a]
        else:
            bell_theta = soft_bellman_model_expected(s, a, probs_learned, r_theta_param, q_table, gamma)
        current_q_value_theta = q_table[s, a]
        total_loss_theta = total_loss_theta + (current_q_value_theta - bell_theta) ** 2
    l_theta = total_loss_theta / float(n_meta_iterations)
    grad_theta = torch.autograd.grad(l_theta, q_table, create_graph=True, retain_graph=True)[0]

    # VJP: map grad_theta (q-space) to param space, contracted with grad_true
    final_grad_products = torch.autograd.grad(
        outputs=grad_theta,
        inputs=[p_alpha_param, r_theta_param],
        grad_outputs=grad_true,
        retain_graph=False,
        allow_unused=True,
    )
    v0 = final_grad_products[0] if (final_grad_products is not None and final_grad_products[0] is not None) else torch.zeros_like(p_alpha_param)
    v1 = final_grad_products[1] if (final_grad_products is not None and final_grad_products[1] is not None) else torch.zeros_like(r_theta_param)

    # assign gradients (negative from IFT) and step
    p_alpha_param.grad = -v0
    r_theta_param.grad = -v1
    optimizer.step()

# --------------------------
# Runner that acts on a Gymnasium Env
# --------------------------
def train_omd_env(
    env,                    # gymnasium.Env (Discrete obs)
    q_table,                # torch.Tensor [S, A]
    target_q_table,         # torch.Tensor [S, A]
    probs_theta,            # torch.Tensor [S, A, S]
    rewards_theta,          # torch.Tensor [S, A]
    max_iterations=1000,
    K=10,
    inner_lr=0.01,
    meta_lr=0.01,
    tau=0.001,
    n_meta_iterations=100,
    gamma=0.5,
    seed=8675309,
    device=torch.device("cpu"),
):
    """
    Train OMD on a gymnasium env. Assumes Discrete observation_space and action_space.
    Returns: q_table.detach(), rewards_theta.detach(), soft(probs_theta).detach(), replay
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    S = env.observation_space.n
    A = env.action_space.n
    states = list(range(S))
    actions = list(range(A))

    replay = SimpleReplay(capacity=10000)

    # Gymnasium reset: returns (obs, info)
    obs, _ = env.reset()
    done = False

    for it in range(max_iterations):
        # action selection from softmax Q (row for current obs)
        with torch.no_grad():
            qvals = q_table[obs, :].cpu()
            action_probs = F.softmax(qvals, dim=0).numpy()
            action_probs /= action_probs.sum()
        a = int(np.random.choice(actions, p=action_probs))

        # Gymnasium step: returns (next_obs, reward, terminated, truncated, info)
        next_obs, reward, terminated, truncated, info = env.step(a)
        done = bool(terminated or truncated)

        replay.add(int(obs), int(a), float(reward), int(next_obs), bool(done))

        # Non-differentiable in-place inner updates for acting
        probs_learned_now = F.softmax(probs_theta, dim=-1)
        for _ in range(K):
            if len(replay) == 0:
                break
            s, aa, r, ns, dflag = replay.sample(1)[0]
            if dflag:
                q_bell = rewards_theta[s, aa].detach()
            else:
                q_bell = soft_bellman_model_expected(s, aa, probs_learned_now, rewards_theta, target_q_table, gamma)
            with torch.no_grad():
                q_table[s, aa] -= inner_lr * 2.0 * (q_table[s, aa] - q_bell.detach())

        # EMA update for target Q
        update_q_table_ema(q_table, target_q_table, tau)

        # Differentiable unroll (q_inner) for meta-update
        q_inner = q_table.clone().detach().requires_grad_(True)
        for _ in range(K):
            if len(replay) == 0:
                break
            s, aa, r, ns, dflag = replay.sample(1)[0]
            q_inner = update_q_step_differentiable(q_inner, s, aa, probs_theta, rewards_theta, gamma, inner_lr)

        # Meta-update of model params via VJP
        update_theta_vjp(
            rewards_theta,
            probs_theta,
            states,
            actions,
            q_inner,
            target_q_table,
            gamma,
            replay,
            n_meta_iterations,
            meta_lr,
            device,
        )

        obs = next_obs
        if done:
            obs, _ = env.reset()
            done = False

        if (it + 1) % max(1, max_iterations // 10) == 0:
            print(f"iter {it+1}/{max_iterations}, replay size {len(replay)}")

    return q_table.detach(), rewards_theta.detach(), F.softmax(probs_theta, dim=-1).detach(), replay

# --------------------------
# Minimal example run (FrozenLake-v1 4x4)
# --------------------------

# Create a Gymnasium environment (Discrete observations)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

S = env.observation_space.n
A = env.action_space.n
device = torch.device("cpu")

# Initialize tabular params (requires_grad where appropriate)
q_table = torch.rand(S, A, requires_grad=True, device=device)
target_q_table = q_table.clone().detach()
rewards_theta = torch.rand(S, A, requires_grad=True, device=device)
probs_theta = torch.rand(S, A, S, requires_grad=True, device=device)

q_learned, r_learned, p_learned, replay = train_omd_env(
    env,
    q_table,
    target_q_table,
    probs_theta,
    rewards_theta,
    max_iterations=500,
    K=4,
    inner_lr=0.05,
    meta_lr=0.01,
    tau=0.01,
    n_meta_iterations=20,
    gamma=0.95,
    seed=42,
    device=device,
)

print("Done. Replay size:", len(replay))
print("Learned rewards (sample):\n", r_learned[:4, :])
print("Learned P (sample slice):\n", p_learned[:2, :2, :5])
