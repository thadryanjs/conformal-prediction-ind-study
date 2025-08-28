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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import torch
import torch.nn.functional as F
from scipy import stats
import os

# %% [code]
#| default_exp src.rl.omd
# simple config
config = {
    # general
    "re-run": True,
    "env_params": {
        "env_name": "FrozenLake-v1",
        "env_kwargs": {
            "map_name": "4x4",
            "is_slippery": False
        }
    },

    "n_seeds": 3,

    # training / algorithm
    "omd_params": {
        "max_iterations": 1200,
        "K": 4,
        "inner_lr": 0.05,
        "meta_lr": 0.01,
        "tau": 0.01,
        "n_meta_iterations": 30,
        "gamma": 0.95,
    },


    # replay / evaluation
    "replay_capacity": 50000,
    "max_iterations": 1200,
    "eval_episodes": 500,

    # device / runtime
    "device": "cpu",

    # output paths
    "out_dir": "outputs/decision-focused-learning",
    "csv_name": "omd-vs-empirical-frozenlake.csv",
    "boxplot_name": "boxplot.png",

    # model-specific options
    "model_params": {
        "probs_init": "random",   # "random" or "uniform"
        "reward_init": "random",  # "random" or "zeros"
        "use_tabular_transitions": True,
    },

    # plotting / display
    "plot_dpi": 200,
    "plot_show": False,  # set True to call plt.show() in interactive use
}


# -------------------------
# SimpleReplay
# -------------------------
#| export
class SimpleReplay:
    """Tiny FIFO replay buffer storing (s,a,r,s',done)."""

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


# -------------------------
# Bellman helpers (with brief math comments)
# -------------------------
#| export
def soft_bellman_model_expected(s, a, probs_model, rewards_model, q_table, gamma):
    """
    Model-based soft Bellman expected target:
      B_θ Q(s,a) = r_θ(s,a) + γ Σ_{s'} p_θ(s'|s,a) log Σ_{a'} exp Q(s',a')
    """
    r_theta = rewards_model[s, a]
    p_s_prime = probs_model[s, a, :]
    next_soft_v = torch.logsumexp(q_table, dim=1)
    expected_future = torch.sum(p_s_prime * next_soft_v)
    return r_theta + gamma * expected_future


#| export
def soft_bellman_buffer_target(
    ds, da, reward_from_buffer, next_state_from_buffer, done_flag, target_q_table, gamma
):
    """
    Buffer-based target:
      if done: target = reward
      else:   target = reward + γ * logsumexp_a' Q_target(next_state, a')
    """
    if not torch.is_tensor(reward_from_buffer):
        reward = torch.as_tensor(
            reward_from_buffer, dtype=target_q_table.dtype, device=target_q_table.device
        )
    else:
        reward = reward_from_buffer.to(
            device=target_q_table.device, dtype=target_q_table.dtype
        )
    if done_flag:
        return reward
    next_q_values = target_q_table[next_state_from_buffer, :]
    return reward + gamma * torch.logsumexp(next_q_values, dim=0)


#| export
def update_q_step_differentiable(
    q_table, ds, da, probs_theta, rewards_theta, gamma, inner_lr
):
    """
    Differentiable inner-loop update:
    loss = ( Q(ds,da) - B_θ Q(ds,da) )^2
    q_new = q - inner_lr * grad_q loss   (create_graph=True to allow higher-order grads)
    """
    probs_learned = F.softmax(probs_theta, dim=-1)
    r_theta = rewards_theta[ds, da]
    p_s_prime = probs_learned[ds, da, :]
    next_soft_v = torch.logsumexp(q_table, dim=1)
    q_bellman = r_theta + gamma * torch.sum(p_s_prime * next_soft_v)
    loss = (q_table[ds, da] - q_bellman) ** 2
    grads = torch.autograd.grad(loss, q_table, create_graph=True)[0]
    q_new = q_table - inner_lr * grads
    return q_new


#| export
def update_q_table_ema(q_source, q_target, tau):
    """EMA update: q_target := (1-tau)*q_target + tau*q_source"""
    with torch.no_grad():
        q_target.data.copy_((1.0 - tau) * q_target.data + tau * q_source.data)


# -------------------------
# VJP meta-update (implicit differentiation)
# -------------------------
#| export
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
    """
    Compute outer gradient d L_true / d theta via VJP:
    - grad_true = d L_true / d q (buffer-based Bellman error)
    - grad_theta_in_q = d L_theta / d q (model-based Bellman error)
    - final grads = (d grad_theta_in_q / d theta)^T grad_true
    """
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
    grad_true = torch.autograd.grad(
        l_estimate_true, q_table, create_graph=True, retain_graph=True
    )[0]

    # grad_theta (model-based)
    total_loss_theta = torch.tensor(0.0, device=device)
    probs_learned = F.softmax(p_alpha_param, dim=-1)
    for i in range(n_meta_iterations):
        s, a, r, ns, done = replay.sample(1)[0]
        if done:
            bell_theta = r_theta_param[s, a]
        else:
            bell_theta = soft_bellman_model_expected(
                s, a, probs_learned, r_theta_param, q_table, gamma
            )
        current_q_value_theta = q_table[s, a]
        total_loss_theta = total_loss_theta + (current_q_value_theta - bell_theta) ** 2
    l_theta = total_loss_theta / float(n_meta_iterations)
    grad_theta = torch.autograd.grad(
        l_theta, q_table, create_graph=True, retain_graph=True
    )[0]

    # VJP: map grad_theta (q-space) -> param space, contracted with grad_true
    final_grad_products = torch.autograd.grad(
        outputs=grad_theta,
        inputs=[p_alpha_param, r_theta_param],
        grad_outputs=grad_true,
        retain_graph=False,
        allow_unused=True,
    )
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

    p_alpha_param.grad = -v0
    r_theta_param.grad = -v1
    optimizer.step()


# -------------------------
# Inner-loop in-place acting updates
# -------------------------
#| export
def update_q_table_inplace_for_acting(
    q_table, target_q_table, probs_theta, rewards_theta, replay, inner_lr, K, gamma
):
    """Fast non-differentiable in-place updates used for acting (Dyna-style)."""
    probs_learned_now = F.softmax(probs_theta, dim=-1)
    for _ in range(K):
        if len(replay) == 0:
            break
        s, aa, r, ns, dflag = replay.sample(1)[0]
        if dflag:
            q_bell = rewards_theta[s, aa].detach()
        else:
            q_bell = soft_bellman_model_expected(
                s, aa, probs_learned_now, rewards_theta, target_q_table, gamma
            )
        with torch.no_grad():
            q_table[s, aa] -= inner_lr * 2.0 * (q_table[s, aa] - q_bell.detach())


# -------------------------
# Soft value iteration (for extracting soft policy)
# -------------------------
#| export
def soft_value_iteration(probs, rewards, gamma=0.95, tol=1e-9, max_iters=10000):
    """Solve soft value iteration: Q(s,a) = r(s,a) + γ Σ_{s'} p(s'|s,a) V(s'); V(s)=logsumexp_a Q(s,a)."""
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


# -------------------------
# Empirical model fit from replay
# -------------------------
#| export
def fit_empirical_model_from_replay(replay, S, A):
    """
    Fit a simple empirical (MLE) tabular model from transitions in replay.
    Returns tensors P_hat [S,A,S] and R_hat [S,A].
    """
    counts = np.zeros((S, A, S), dtype=float)
    rew_sum = np.zeros((S, A), dtype=float)
    act_counts = np.zeros((S, A), dtype=float)
    for s, a, r, ns, done in replay.buf:
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
    return torch.as_tensor(P_hat, dtype=torch.float32), torch.as_tensor(
        R_hat, dtype=torch.float32
    )


# %% [code]
# | export
def compute_soft_policy_from_model(P_torch, R_torch, gamma=0.99):
    _, _, soft_pi, _ = soft_value_iteration(P_torch, R_torch, gamma=gamma)
    return soft_pi


# -------------------------
# Policy evaluation (rollouts)
# -------------------------
#| export
def evaluate_policy(env, policy_probs, n_episodes=500, seed=0):
    """
    Roll out the stationary soft policy (policy_probs: array/tensor [S,A]) for n_episodes
    and return mean and std of episodic returns.
    """
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


# -------------------------
# Per-state KL
# -------------------------
#| export
def per_state_kl(p_true, p_est, eps=1e-12):
    p_true = p_true.clamp(min=eps)
    p_est = p_est.clamp(min=eps)
    return torch.sum(p_true * (torch.log(p_true) - torch.log(p_est)), dim=1)


# -------------------------
# Main training function (OMD) on Gymnasium env (FrozenLake)
# -------------------------
#| export
def train_omd(
    env,
    seed,
    max_iterations=1200,
    K=4,
    inner_lr=0.05,
    meta_lr=0.01,
    tau=0.01,
    n_meta_iterations=30,
    gamma=0.95,
    device=torch.device("cpu"),
):
    """
    Train OMD on a gymnasium env with Discrete observations and actions.
    Returns learned model (P,R) and the replay buffer.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    S = env.observation_space.n
    A = env.action_space.n

    # initialize model params (probs logits and rewards)
    rewards_theta = torch.rand(S, A, requires_grad=True, device=device)
    probs_theta = torch.rand(S, A, S, requires_grad=True, device=device)

    # live Q and EMA target
    q_table = torch.rand(S, A, requires_grad=False, device=device)
    target_q_table = q_table.clone().detach()

    replay = SimpleReplay(capacity=10000)

    obs, _ = env.reset()
    done = False

    for it in range(max_iterations):
        # select action from softmax over target Q for stability
        with torch.no_grad():
            qvals = target_q_table[obs, :].cpu()
            action_probs = F.softmax(qvals, dim=0).numpy()
            action_probs /= action_probs.sum()
        a = int(np.random.choice(range(A), p=action_probs))

        next_obs, reward, terminated, truncated, info = env.step(a)
        done = bool(terminated or truncated)
        replay.add(int(obs), int(a), float(reward), int(next_obs), bool(done))

        # in-place inner updates for acting (non-diff)
        update_q_table_inplace_for_acting(
            q_table,
            target_q_table,
            probs_theta,
            rewards_theta,
            replay,
            inner_lr,
            K,
            gamma,
        )

        # EMA update
        update_q_table_ema(q_table, target_q_table, tau)

        # differentiable unroll
        q_inner = q_table.clone().detach().requires_grad_(True)
        for _ in range(K):
            if len(replay) == 0:
                break
            s, aa, r, ns, dflag = replay.sample(1)[0]
            q_inner = update_q_step_differentiable(
                q_inner, s, aa, probs_theta, rewards_theta, gamma, inner_lr
            )

        # outer update
        update_theta_vjp(
            rewards_theta,
            probs_theta,
            list(range(S)),
            list(range(A)),
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

    learned_P = F.softmax(probs_theta, dim=-1).detach().cpu()
    learned_R = rewards_theta.detach().cpu()
    return learned_P, learned_R, replay


# -------------------------
# Batch-run and evaluation (uses above functions)
# -------------------------
#| export
def batch_run(
    env_params,
    omd_params,
    n_seeds,
    eval_episodes,
):
    rows = []
    for i in range(n_seeds):
        seed = 1000 + i
        env = gym.make(env_params["env_name"], **env_params["env_kwargs"])
        learned_P, learned_R, replay = train_omd(
            env, seed, **omd_params
        )
        pi_learned = compute_soft_policy_from_model(learned_P, learned_R, gamma=0.95)
        S = env.observation_space.n
        A = env.action_space.n
        P_hat, R_hat = fit_empirical_model_from_replay(replay, S, A)
        pi_emp = compute_soft_policy_from_model(P_hat, R_hat, gamma=0.95)
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
        mean_learn, std_learn = evaluate_policy(
            env, pi_learned, n_episodes=eval_episodes, seed=seed
        )
        mean_emp, std_emp = evaluate_policy(
            env, pi_emp, n_episodes=eval_episodes, seed=seed
        )
        mean_true, std_true = (None, None)
        if pi_true is not None:
            mean_true, std_true = evaluate_policy(
                env, pi_true, n_episodes=eval_episodes, seed=seed
            )
        kl_learn_mean = kl_emp_mean = None
        if pi_true is not None:
            kl_learn = per_state_kl(pi_true, pi_learned)
            kl_emp = per_state_kl(pi_true, pi_emp)
            kl_learn_mean = float(kl_learn.mean().item())
            kl_emp_mean = float(kl_emp.mean().item())
        rows.append(
            {
                "seed": int(seed),
                "mean_learn": float(mean_learn),
                "std_learn": float(std_learn),
                "mean_emp": float(mean_emp),
                "std_emp": float(std_emp),
                "mean_true": float(mean_true) if mean_true is not None else None,
                "std_true": float(std_true) if std_true is not None else None,
                "kl_learn_mean": kl_learn_mean,
                "kl_emp_mean": kl_emp_mean,
                "replay_size": len(replay),
            }
        )
        env.close()
        print(
            f"Finished seed {i+1}/{n_seeds}: mean_learn={mean_learn:.4f}, mean_emp={mean_emp:.4f}, replay={len(replay)}"
        )
    df = pd.DataFrame(rows)
    return df


# main run
out_dir = config["out_dir"]
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "omd-vs-empirical-frozenlake.csv")


if config["re-run"]:
    df = batch_run(
        env_params=config["env_params"],
        omd_params=config["omd_params"],
        # I can ** these too
        n_seeds=config["n_seeds"],
        eval_episodes=config["eval_episodes"],
    )
    df.to_csv(csv_path, index=False)
else:
    df = pd.read_csv(csv_path)

# compare methods: paired t-test
learned_means = df["mean_learn"]
empirical_means = df["mean_emp"]
t_stat, p_value = stats.ttest_rel(learned_means, empirical_means)
diffs = learned_means - empirical_means

mean_diff = diffs.mean()

se_diff = diffs.std(ddof=1) / np.sqrt(len(diffs))
alpha = 0.05
df_dof = len(diffs) - 1
t_crit = stats.t.ppf(1 - alpha / 2, df_dof)
ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff

print(f"paired t-test: t = {t_stat:.4f}, p = {p_value:.4e}")
print(
    f"mean difference = {mean_diff:.6f} ± {t_crit * se_diff:.6f} (95% CI = [{ci_lower:.6f}, {ci_upper:.6f}])"
)

# simple boxplot + dots
plt.figure(figsize=(6, 4))
sns.set_style("white")
ax = sns.boxplot(
    data=[learned_means, empirical_means],
    palette=["#4c72b0", "#dd8452"],
    showfliers=False,
)
sns.stripplot(
    data=[learned_means, empirical_means],
    color="k",
    size=6,
    jitter=0.12,
    dodge=True,
    alpha=0.8,
)
p_text = f"paired t-test: p = {p_value:.3g}"
md_text = f"mean diff = {mean_diff:.4f}"
plt.text(
    0.5,
    max(np.max(learned_means), np.max(empirical_means)) * 1.05,
    p_text + "  |  " + md_text,
    ha="center",
    va="bottom",
    fontsize=10,
)
plt.xticks([0, 1], ["learned_means", "empirical_means"])
plt.ylabel("Mean episodic return")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "omdboxplot.png"), dpi=300)
