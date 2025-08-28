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

# omd_gymnasium_evaluate.py
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
# Evaluation helpers
# --------------------------
def soft_value_iteration(probs, rewards, gamma=0.95, tol=1e-9, max_iters=10000):
    S, A, _ = probs.shape
    V = torch.zeros(S, dtype=rewards.dtype, device=rewards.device)
    Q = torch.empty((S, A), dtype=rewards.dtype, device=rewards.device)
    for i in range(max_iters):
        V_prev = V
        for s in range(S):
            for a in range(A):
                Q[s, a] = rewards[s, a] + gamma * torch.dot(probs[s, a], V_prev)
        V = torch.logsumexp(Q, dim=1)
        if torch.max(torch.abs(V - V_prev)) < tol:
            break
    return V, Q, F.softmax(Q, dim=1), i + 1

def fit_empirical_model_from_replay(replay, S, A):
    counts = np.zeros((S, A, S), dtype=float)
    rew_sum = np.zeros((S, A), dtype=float)
    act_counts = np.zeros((S, A), dtype=float)
    for (s, a, r, ns, done) in replay.buf:
        counts[s, a, ns] += 1.0
        rew_sum[s, a] += r
        act_counts[s, a] += 1.0
    P_hat = np.zeros_like(counts)
    for s in range(S):
        for a in range(A):
            total = counts[s, a].sum()
            if total > 0:
                P_hat[s, a] = counts[s, a] / total
            else:
                P_hat[s, a] = np.ones(S) / float(S)
    R_hat = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            if act_counts[s, a] > 0:
                R_hat[s, a] = rew_sum[s, a] / act_counts[s, a]
            else:
                R_hat[s, a] = 0.0
    return torch.as_tensor(P_hat, dtype=torch.float32), torch.as_tensor(R_hat, dtype=torch.float32)

def compute_soft_policy_from_model(P_torch, R_torch, gamma=0.99):
    _, _, soft_pi, _ = soft_value_iteration(P_torch, R_torch, gamma=gamma)
    return soft_pi

def per_state_kl(p_true, p_est, eps=1e-12):
    p_true = p_true.clamp(min=eps)
    p_est = p_est.clamp(min=eps)
    return torch.sum(p_true * (torch.log(p_true) - torch.log(p_est)), dim=1)

def evaluate_policy(env, policy_probs, n_episodes=200, seed=0):
    rng = np.random.RandomState(seed)
    total_returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            probs = policy_probs[obs]
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
            probs = np.array(probs, dtype=float)
            probs = probs / (probs.sum() + 1e-12)
            a = int(rng.choice(len(probs), p=probs))
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            obs = next_obs
        total_returns.append(ep_ret)
    return float(np.mean(total_returns)), float(np.std(total_returns))

# --------------------------
# Main: train, fit baseline, compute policies, evaluate
# --------------------------
if __name__ == "__main__":
    # Choose environment: FrozenLake 4x4 (Discrete). Set is_slippery=False for deterministic.
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

    S = env.observation_space.n
    A = env.action_space.n
    device = torch.device("cpu")

    # Initialize tabular params (learnable)
    q_table = torch.rand(S, A, requires_grad=True, device=device)
    target_q_table = q_table.clone().detach()
    rewards_theta = torch.rand(S, A, requires_grad=True, device=device)
    probs_theta = torch.rand(S, A, S, requires_grad=True, device=device)

    # Train OMD on env
    learned_q, learned_R, learned_P, replay = train_omd_env(
        env,
        q_table,
        target_q_table,
        probs_theta,
        rewards_theta,
        max_iterations=1500,
        K=4,
        inner_lr=0.05,
        meta_lr=0.01,
        tau=0.01,
        n_meta_iterations=30,
        gamma=0.95,
        seed=42,
        device=device,
    )

    print("\nTraining completed. Replay size:", len(replay))

    # Learned model -> soft policy
    pi_learned = compute_soft_policy_from_model(learned_P, learned_R, gamma=0.95)

    # Empirical model fit from replay -> soft policy
    P_hat, R_hat = fit_empirical_model_from_replay(replay, S, A)
    pi_empirical = compute_soft_policy_from_model(P_hat, R_hat, gamma=0.95)

    # True model (if available) -> soft policy
    pi_true = None
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "P"):
        P_true_arr = np.zeros((S, A, S), dtype=float)
        R_true_arr = np.zeros((S, A), dtype=float)
        for s in range(S):
            for a in range(A):
                for prob, ns, rew, done in env.unwrapped.P[s][a]:
                    P_true_arr[s, a, ns] += prob
                    R_true_arr[s, a] = rew
        P_true = torch.as_tensor(P_true_arr, dtype=torch.float32)
        R_true = torch.as_tensor(R_true_arr, dtype=torch.float32)
        _, _, pi_true, _ = soft_value_iteration(P_true, R_true, gamma=0.95)

    # Compute per-state KLs if true policy exists
    if pi_true is not None:
        kl_learn = per_state_kl(pi_true, pi_learned)
        kl_emp = per_state_kl(pi_true, pi_empirical)
        print("\nPer-state KL (true || learned):", kl_learn.numpy())
        print("Per-state KL (true || empirical):", kl_emp.numpy())

    # Evaluate policies by rollout
    mean_learn, std_learn = evaluate_policy(env, pi_learned, n_episodes=500, seed=123)
    mean_emp, std_emp = evaluate_policy(env, pi_empirical, n_episodes=500, seed=123)
    print(f"\nLearned policy rollout mean return: {mean_learn:.4f} +- {std_learn:.4f}")
    print(f"Empirical policy rollout mean return: {mean_emp:.4f} +- {std_emp:.4f}")

    if pi_true is not None:
        mean_true, std_true = evaluate_policy(env, pi_true, n_episodes=500, seed=123)
        print(f"True policy rollout mean return: {mean_true:.4f} +- {std_true:.4f}")


# here is the pseudo code I want to implement
seeds = []
omd_results = []
mle_results = []

for i in range(30):
    # set a seed
    env.seed(i)
    seeds.append(i)

    # train omd
    omd_results.append(train_omd_env(
        env,
        q_table,
        target_q_table,
        probs_theta,
        rewards_theta,
        max_iterations=1500,
        K=4,
        inner_lr=0.05,
        meta_lr=0.01,
        tau=0.01,
        n_meta_iterations=30,
        gamma=0.95,
        seed=42,
        device=device,
    ))

    # train mle
    mle_results.append(train_mle_env(
        env,
        q_table,
        target_q_table,
        probs_theta,
        rewards_theta,
        max_iterations=1500,
        K=4,
        inner_lr=0.05,
        meta_lr=0.01,
        tau=0.01,
        n_meta_iterations=30,
        gamma=0.95,
        seed=42,
        device=device,
    ))

df_results = pd.DataFrame({
    "seed": seeds,
    "omd_q": [r[0] for r in omd_results],
    "omd_R": [r[1] for r in omd_results],
    "omd_P": [r[2] for r in omd_results],
    "omd_replay": [r[3] for r in omd_results],
    "mle_q": [r[0] for r in mle_results],
    "mle_R": [r[1] for r in mle_results],
    "mle_P": [r[2] for r in mle_results],
    "mle_replay": [r[3] for r in mle_results],
})

