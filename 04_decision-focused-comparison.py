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


# omd_singlefile_gym.py
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F

# -------------------------
# Gym-idiomatic TabularMDPEnv
# -------------------------
class TabularMDPEnv(gym.Env):
    def __init__(self, P, R, terminal_states=None, max_episode_steps=None, seed=None):
        super().__init__()
        P = np.asarray(P, dtype=float)
        R = np.asarray(R, dtype=float)
        assert P.ndim == 3 and R.ndim >= 2
        S, A, S2 = P.shape
        assert S == S2 and R.shape[:2] == (S, A)
        # normalize P rows
        for s in range(S):
            for a in range(A):
                row = P[s, a].astype(float)
                sm = row.sum()
                if sm <= 0:
                    P[s, a] = np.ones(S) / float(S)
                else:
                    P[s, a] = row / sm

        self.P = P
        self.R = R
        self.S = S
        self.A = A

        self.observation_space = spaces.Discrete(S)
        self.action_space = spaces.Discrete(A)

        self.terminal_states = np.zeros(S, dtype=bool) if terminal_states is None else np.asarray(terminal_states, dtype=bool)
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)

        self._rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        self.seed(seed)
        self._state = 0
        self._elapsed = 0

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        self._rng = np.random.RandomState(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        non_term = np.where(~self.terminal_states)[0]
        self._state = int(self._rng.choice(non_term)) if len(non_term) > 0 else 0
        self._elapsed = 0
        return int(self._state)

    def step(self, action):
        assert self.action_space.contains(action)
        s = int(self._state)
        if self.terminal_states[s]:
            r = float(self.R[s, action])
            return s, r, True, {"terminal_state": True}
        probs = self.P[s, action]
        ns = int(self._rng.choice(self.S, p=probs))
        r = float(self.R[s, action])
        self._state = ns
        self._elapsed += 1
        done = bool(self.terminal_states[ns]) or (self.max_episode_steps is not None and self._elapsed >= self.max_episode_steps)
        return ns, r, done, {}

    def render(self, mode="human"):
        out = f"State: {self._state}"
        if mode == "human":
            print(out)
        else:
            return out

    def close(self):
        pass

# -------------------------
# Simple replay buffer
# -------------------------
class SimpleReplay:
    def __init__(self, capacity=10000):
        self.buf = []
        self.capacity = capacity
    def add(self, s, a, r, ns, done):
        if len(self.buf) >= self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, ns, done))
    def sample(self, n=1):
        assert len(self.buf) > 0
        idx = np.random.choice(len(self.buf), size=n)
        return [self.buf[i] for i in idx]
    def __len__(self):
        return len(self.buf)

# -------------------------
# Core functions (minimal)
# -------------------------
def get_softmax_policies(states, actions, q_table):
    res = {}
    with torch.no_grad():
        for s in states:
            v = q_table[s].cpu()
            p = F.softmax(v, dim=0).numpy()
            p /= p.sum()
            res[s] = {a: float(p[a]) for a in actions}
    return res

def soft_bellman_model_expected(s, a, probs_model, rewards_model, q_table, gamma):
    r = rewards_model[s, a]
    p = probs_model[s, a, :]
    if isinstance(p, np.ndarray):
        p = torch.as_tensor(p, dtype=q_table.dtype, device=q_table.device)
    next_soft_v = torch.logsumexp(q_table, dim=1)
    expected = torch.sum(p * next_soft_v)
    return r + gamma * expected

def soft_bellman_buffer_target(ds, da, reward_from_buffer, next_state_from_buffer, done_flag, target_q_table, gamma):
    if not torch.is_tensor(reward_from_buffer):
        reward = torch.as_tensor(reward_from_buffer, dtype=target_q_table.dtype, device=target_q_table.device)
    else:
        reward = reward_from_buffer.to(device=target_q_table.device, dtype=target_q_table.dtype)
    if done_flag:
        return reward
    next_q = target_q_table[next_state_from_buffer]
    return reward + gamma * torch.logsumexp(next_q, dim=0)

def update_q_step_differentiable(q_table, ds, da, probs_theta, rewards_theta, gamma, inner_lr):
    probs_learned = F.softmax(probs_theta, dim=-1)
    r_theta = rewards_theta[ds, da]
    p_s_prime = probs_learned[ds, da, :]
    next_soft_v = torch.logsumexp(q_table, dim=1)
    expected_future = torch.sum(p_s_prime * next_soft_v)
    q_bellman = r_theta + gamma * expected_future
    loss = (q_table[ds, da] - q_bellman) ** 2
    grads = torch.autograd.grad(loss, q_table, create_graph=True)[0]
    return q_table - inner_lr * grads

def update_q_table_ema(q_source, q_target, tau):
    with torch.no_grad():
        q_target.data.copy_((1.0 - tau) * q_target.data + tau * q_source.data)

# VJP meta-update
def update_theta_vjp(r_theta_param, p_alpha_param, q_table, target_q_table, gamma, replay, n_meta_iterations, lr, device):
    if len(replay) == 0:
        return
    opt = torch.optim.Adam([r_theta_param, p_alpha_param], lr=lr)
    opt.zero_grad()
    total_loss_true = torch.tensor(0.0, device=device)
    for _ in range(n_meta_iterations):
        s, a, r, ns, done = replay.sample(1)[0]
        bell = soft_bellman_buffer_target(s, a, r, ns, done, target_q_table, gamma)
        total_loss_true = total_loss_true + (q_table[s, a] - bell.detach()) ** 2
    l_true = total_loss_true / float(n_meta_iterations)
    grad_true = torch.autograd.grad(l_true, q_table, create_graph=True, retain_graph=True)[0]

    total_loss_theta = torch.tensor(0.0, device=device)
    probs_learned = F.softmax(p_alpha_param, dim=-1)
    for _ in range(n_meta_iterations):
        s, a, r, ns, done = replay.sample(1)[0]
        if done:
            bell_theta = r_theta_param[s, a]
        else:
            bell_theta = soft_bellman_model_expected(s, a, probs_learned, r_theta_param, q_table, gamma)
        total_loss_theta = total_loss_theta + (q_table[s, a] - bell_theta) ** 2
    l_theta = total_loss_theta / float(n_meta_iterations)
    grad_theta = torch.autograd.grad(l_theta, q_table, create_graph=True, retain_graph=True)[0]

    final = torch.autograd.grad(outputs=grad_theta, inputs=[p_alpha_param, r_theta_param], grad_outputs=grad_true, allow_unused=True, retain_graph=False)
    v0 = final[0] if (final is not None and final[0] is not None) else torch.zeros_like(p_alpha_param)
    v1 = final[1] if (final is not None and final[1] is not None) else torch.zeros_like(r_theta_param)
    p_alpha_param.grad = -v0
    r_theta_param.grad = -v1
    opt.step()

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

# -------------------------
# Trainer (uses gym.Env)
# -------------------------
def train_omd_on_env(env, seed=0, device=torch.device("cpu")):
    np.random.seed(seed)
    torch.manual_seed(seed)
    S = env.observation_space.n
    A = env.action_space.n
    states = list(range(S))
    actions = list(range(A))

    # If env provides P/R (e.g., FrozenLake) capture them for diagnostics
    P_true = None
    R_true = None
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "P"):
        P_true = np.zeros((S, A, S), dtype=float)
        R_true = np.zeros((S, A), dtype=float)
        for s in range(S):
            for a in range(A):
                for prob, ns, rew, done in env.unwrapped.P[s][a]:
                    P_true[s, a, ns] += prob
                    R_true[s, a] = rew

    # initialize params
    q_table = torch.rand(S, A, requires_grad=True, device=device)
    target_q_table = q_table.clone().detach()
    rewards_theta = torch.rand(S, A, requires_grad=True, device=device)
    probs_theta = torch.rand(S, A, S, requires_grad=True, device=device)

    replay = SimpleReplay(capacity=5000)
    max_iters = 1200
    K = 4
    inner_lr = 0.1
    meta_lr = 0.01
    tau = 0.05
    n_meta_iterations = 20
    gamma = 0.99

    obs = env.reset()
    done = False
    for it in range(max_iters):
        policies = get_softmax_policies(states, actions, q_table)
        pvals = np.array([policies[obs][a] for a in actions], dtype=float)
        pvals /= pvals.sum()
        a = int(np.random.choice(actions, p=pvals))

        next_obs, reward, done, info = env.step(a)
        replay.add(obs, a, float(reward), next_obs, bool(done))

        # non-diff inner updates used for acting
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

        # differentiable unroll
        q_inner = q_table.clone().detach().requires_grad_(True)
        for _ in range(K):
            if len(replay) == 0:
                break
            s, aa, r, ns, dflag = replay.sample(1)[0]
            q_inner = update_q_step_differentiable(q_inner, s, aa, probs_theta, rewards_theta, gamma, inner_lr)

        # meta-update of model parameters
        update_theta_vjp(rewards_theta, probs_theta, q_inner, target_q_table, gamma, replay, n_meta_iterations, meta_lr, device)

        obs = next_obs
        if done:
            obs = env.reset()
            done = False

        if (it + 1) % 300 == 0:
            print(f"iter {it+1}/{max_iters}, replay size {len(replay)}")

    learned_P = F.softmax(probs_theta, dim=-1).detach().cpu()
    learned_R = rewards_theta.detach().cpu()
    return learned_P, learned_R, q_table.detach().cpu(), replay, P_true, R_true

# -------------------------
# Example run: FrozenLake (4x4) and comparison to empirical baseline
# -------------------------
if __name__ == "__main__":
    # Choose environment: FrozenLake (idiomatic Gym env)
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    learned_P, learned_R, q_table, replay, P_true, R_true = train_omd_on_env(env, seed=42, device=torch.device("cpu"))

    print("\n--- Results ---")
    V_learn, Q_learn, pi_learn, _ = soft_value_iteration(learned_P, learned_R, gamma=0.99)
    print("Learned soft policy (from learned model):")
    print(pi_learn.numpy())

    P_hat, R_hat = fit_empirical_model_from_replay(replay, env.observation_space.n, env.action_space.n)
    V_hat, Q_hat, pi_hat, _ = soft_value_iteration(P_hat, R_hat, gamma=0.99)
    print("\nEmpirical-model soft policy (from replay counts):")
    print(pi_hat.numpy())

    if P_true is not None:
        P_true_t = torch.as_tensor(P_true, dtype=torch.float32)
        R_true_t = torch.as_tensor(R_true, dtype=torch.float32)
        V_true, Q_true, pi_true, _ = soft_value_iteration(P_true_t, R_true_t, gamma=0.99)
        print("\nTrue soft policy (from env.unwrapped.P):")
        print(pi_true.numpy())
        # per-state KL
        kl = (pi_true * (pi_true.log() - pi_learn.log())).sum(dim=1)
        print("\nPer-state KL(true || learned):", kl.numpy())

    print("\nReplay size:", len(replay))
