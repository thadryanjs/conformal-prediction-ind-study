# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [code]
import torch
from torch import Tensor
from typing import Tuple, Any


# %% [code]
# config dict
cfg = {
    "s_max": 10,
    "a_max": 5,
    "h": 10,
    "hold_cost": 1.0,
    "shortage_cost": 10.0,
    "order_cost": 2.0,
    "demand_lambda": 2.0,
    "seed": 0,
    "initial_s": 2,
    "n_episodes": 1000,
    "device": "cpu",  # or 'cuda'
}

# device and generator
device = torch.device(cfg["device"])
gen = torch.Generator(device=device)
# int: let's pyrefly know the type
gen.manual_seed(int(cfg["seed"]))


# %% [code]
# poisson sampling
def sample_demand(
    size: tuple,
    demand_lambda: float,
    s_max: int,
    a_max: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    lam = torch.full(size, demand_lambda, device=device)
    d = torch.poisson(lam, generator=generator)
    d = torch.clamp(d, max=(s_max + a_max))
    return d.to(torch.long)


# test the function
d = sample_demand((), 2.0, 10, 5, device, gen)
print(d)


# %% [code]
# moves us one era forward
def step(
    s,
    a,
    demand_lambda,
    s_max,
    a_max,
    hold_cost,
    shortage_cost,
    order_cost,
    device,
    generator,
    # return a tuple of three tensors
) -> Tuple[Tensor]:
    # the upper limit as we're dealing with two integers
    inventory = s + a
    # () is for 0D tensor ie a sclar of type Tensor
    d = sample_demand((), demand_lambda, s_max, a_max, device, generator)
    # gets whatever sales happen we couldn't fill
    lost_sales = torch.clamp(d - inventory, min=0)
    # makes sure it doesn't exceed the range of the model
    s_next = torch.clamp(inventory - d, min=0)
    # we start by losing holding cost (overhead)
    reward = -(
        # represents the holding cost is overhead for next state
        hold_cost * s_next.to(torch.float32)
        # represents the shortage cost * how many we lost
        + shortage_cost * lost_sales.to(torch.float32)
        # represents the price of action * next action
        + order_cost * a.to(torch.float32)
    )
    return s_next.to(torch.long), reward, d


# test the function specify args
s, r, d = step(
    s=torch.tensor(2, dtype=torch.long, device=device),
    a=torch.tensor(1, dtype=torch.long, device=device),
    demand_lambda=cfg["demand_lambda"],
    s_max=cfg["s_max"],
    a_max=cfg["a_max"],
    hold_cost=cfg["hold_cost"],
    shortage_cost=cfg["shortage_cost"],
    order_cost=cfg["order_cost"],
    device=device,
    generator=gen,
)
print(s, r, d)


# %% [code]
# behaviour policy (returns probs tensor shape (A,)) for exploring actions
"""
The base mase is arbitrary, essentially just saying to start somewere for all actions.
The preffered mass is how much you want to give the preferred action.
The numbers are then normalized, so it's not 60% base and 40% preferred necessarily.
"""
def behaviour_policy_probs(s, a_max,
                           preferred=2, pref_mass=0.6, base_mass=0.05) -> Tensor:
    A = a_max + 1
    probs = torch.full((A,), base_mass, dtype=torch.float32, device=device)
    pref = int(min(max(preferred, 0), a_max))
    probs[pref] += pref_mass
    probs = probs / probs.sum()
    return probs

# test the function
probs = behaviour_policy_probs(s=2, a_max=5, preferred=2, pref_mass=0.6, base_mass=0.05)
print(probs)


# %% [code]
# target policy (deterministic-like)
"""
This is simply for returning the preferred action as a one-hot vector
"""
def target_policy_probs(s, theta, a_max) -> Tensor:
    desired = int(min(max(round(max(0.0, float(theta) - float(s))), 0), a_max))
    A = a_max + 1
    probs = torch.zeros((A,), dtype=torch.float32, device=device)
    probs[desired] = 1.0
    return probs

# test the function
probs = target_policy_probs(s=2, theta=3, a_max=5)
print(probs)


# %% [code]
# sample episode under a policy (policy_fn returns probs tensor)
"""
This takes a policy function, samples the episode, and returns the states, actions, rewards, and probs
"""
def sample_episode(
    initial_s,
    h,
    a_max,
    policy_fn,
    *,
    demand_lambda,
    s_max,
    hold_cost,
    shortage_cost,
    order_cost,
    theta=None,
    generator=None,
):
    s = torch.tensor(initial_s, dtype=torch.long, device=device)
    states = []
    actions = []
    rewards = []
    probs_list = []
    for _ in range(h):
        if theta is None:
            probs = policy_fn(s, a_max)
        else:
            probs = policy_fn(s, theta, a_max)
        action = torch.multinomial(
            probs, num_samples=1, replacement=True, generator=generator
        ).squeeze(0)
        s_next, r, d = step(
            s,
            action,
            demand_lambda,
            s_max,
            a_max,
            hold_cost,
            shortage_cost,
            order_cost,
            device,
            generator,
        )
        states.append(s)
        actions.append(action)
        rewards.append(r)
        probs_list.append(probs)
        s = s_next
    return states, actions, rewards, probs_list


# test the function
states, actions, rewards, probs_list = sample_episode(
    initial_s=2,
    h=5,
    a_max=5,
    policy_fn=behaviour_policy_probs,
    demand_lambda=cfg["demand_lambda"],
    s_max=cfg["s_max"],
    hold_cost=cfg["hold_cost"],
    shortage_cost=cfg["shortage_cost"],
    order_cost=cfg["order_cost"],
    theta=None,
    generator=gen,
)
print(states, actions, rewards, probs_list)


# %% [code]
# collect batch under behaviour policy
"""

"""
def collect_batch(
    n,
    initial_s,
    h,
    a_max,
    behaviour_policy_fn,
    demand_lambda,
    s_max,
    hold_cost,
    shortage_cost,
    order_cost,
    generator,
) -> Tuple[list, Tensor]:
    episodes = []
    returns = torch.zeros((n,), dtype=torch.float32, device=device)
    for i in range(n):
        states, actions, rewards, b_probs = sample_episode(
            initial_s=initial_s,
            h=h,
            a_max=a_max,
            policy_fn=behaviour_policy_fn,
            demand_lambda=demand_lambda,
            s_max=s_max,
            hold_cost=hold_cost,
            shortage_cost=shortage_cost,
            order_cost=order_cost,
            theta=None,
            generator=generator,
        )
        episodes.append((states, actions, rewards, b_probs))
        returns[i] = torch.stack(rewards).sum()
    return episodes, returns

# test the function
episodes, returns = collect_batch(
    cfg["n_episodes"],
    cfg["initial_s"],
    cfg["h"],
    cfg["a_max"],
    behaviour_policy_probs,
    cfg["demand_lambda"],
    cfg["s_max"],
    cfg["hold_cost"],
    cfg["shortage_cost"],
    cfg["order_cost"],
    gen,
)
print(episodes, returns)


# %% [code]
# convert episodes to batched tensors
"""
This is a purely logistical function dealing with types
"""
def episodes_to_tensors(episodes, a_max, device):
    n_eps = len(episodes)
    T = len(episodes[0][0])
    A = a_max + 1
    states = torch.zeros((n_eps, T), dtype=torch.long, device=device)
    actions = torch.zeros((n_eps, T), dtype=torch.long, device=device)
    rewards = torch.zeros((n_eps, T), dtype=torch.float32, device=device)
    b_probs = torch.zeros((n_eps, T, A), dtype=torch.float32, device=device)
    for i, (sts, acts, rws, bps) in enumerate(episodes):
        for t in range(T):
            states[i, t] = sts[t]
            actions[i, t] = acts[t]
            rewards[i, t] = rws[t]
            b_probs[i, t] = bps[t]
    return states, actions, rewards, b_probs

# test the function
states, actions, rewards, b_probs = episodes_to_tensors(
    episodes, cfg["a_max"], device
)
print(states, actions, rewards, b_probs)


# %% [code]
# build target probs tensor
"""
This builds one-hot vectors for the target policy. The states inputs is random, so it doesn't produce symetrical output, ie the location of the 1 doesn't go left to right:

        [[0., 1., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0.],
"""
def build_target_probs_tensor(states, theta, a_max):
    n_eps, T = states.shape
    A = a_max + 1
    pi = torch.zeros((n_eps, T, A), dtype=torch.float32, device=device)
    for i in range(n_eps):
        for t in range(T):
            s = states[i, t]
            raw_desired = float(theta) - float(s)
            desired = int(round(max(0.0, raw_desired)))
            desired = max(0, min(desired, a_max))
            p = torch.zeros((A,), dtype=torch.float32, device=device)
            p[desired] = 1.0
            pi[i, t] = p
    return pi

# test the function
pi = build_target_probs_tensor(states, 3, cfg["a_max"])
print(pi)


# %% [code]
# importance sampling helpers
"""
Ugly tensor manipulation to compute importance sampling weights
"""
def episode_weights_from_probs(pi_probs, b_probs, actions, eps=1e-12):
    # 1) pick π(a_t | s_t) for each taken action -> shape (n_eps, T)
    action_indices = actions.unsqueeze(-1)              # (n_eps, T, 1)
    pi_for_taken = pi_probs.gather(dim=-1, index=action_indices).squeeze(-1)

    # 2) pick b(a_t | s_t) for each taken action -> shape (n_eps, T)
    b_for_taken = b_probs.gather(dim=-1, index=action_indices).squeeze(-1)

    # 3) compute per-step importance ratios (add eps for numerical safety)
    ratios = (pi_for_taken + eps) / (b_for_taken + eps)  # (n_eps, T)

    # 4) work in log-space for numerical stability: sum log ratios over time
    log_ratios = torch.log(ratios)                      # (n_eps, T)
    log_weights = log_ratios.sum(dim=1)                 # (n_eps,)  sum over time

    # 5) exponentiate to get product-of-ratios = episode weight
    weights = torch.exp(log_weights)                    # (n_eps,)

    return weights

# test the function
w = episode_weights_from_probs(pi, b_probs, actions)
print(w)


# %% [code]
"""
Vanilla importance sampling
"""
def ordinary_is(pi_probs, b_probs, actions, returns, eps=1e-12):
    w = episode_weights_from_probs(pi_probs, b_probs, actions, eps=eps)
    est = (w * returns).mean()
    se = (w * returns).std(unbiased=True) / torch.tensor(
        len(returns), dtype=returns.dtype, device=device
    )
    return float(est.item()), float(se.item())

# test the function
episode_returns = rewards.sum(dim=1)
est, se = ordinary_is(pi, b_probs, actions, episode_returns)
print(est, se)


# %% [code]
"""
Weighted importance sampling
"""
def weighted_is(pi_probs, b_probs, actions, returns, eps=1e-12):
    w = episode_weights_from_probs(pi_probs, b_probs, actions, eps=eps)
    wsum = w.sum()
    if float(wsum.item()) == 0.0:
        return 0.0, 0.0
    est = (w * returns).sum() / (wsum + eps)
    se = ((w * returns / (wsum + eps)).std(unbiased=True)) / torch.tensor(
        len(returns), dtype=returns.dtype, device=device
    )
    return float(est.item()), float(se.item())

# test the function
est, se = weighted_is(pi, b_probs, actions, episode_returns)
print(est, se)


# %% [code]
# main
if __name__ == "__main__":
    episodes, returns = collect_batch(
        cfg["n_episodes"],
        cfg["initial_s"],
        cfg["h"],
        cfg["a_max"],
        behaviour_policy_probs,
        cfg["demand_lambda"],
        cfg["s_max"],
        cfg["hold_cost"],
        cfg["shortage_cost"],
        cfg["order_cost"],
        gen,
    )

    print("behaviour empirical mean return (torch):", float(returns.mean().item()))

    states, actions, rewards, b_probs = episodes_to_tensors(
        episodes, cfg["a_max"], device
    )
    returns_from_rewards = rewards.sum(dim=1)

    theta = 4
    pi_probs = build_target_probs_tensor(states, theta, cfg["a_max"])

    o_est, o_se = ordinary_is(pi_probs, b_probs, actions, returns_from_rewards)
    w_est, w_se = weighted_is(pi_probs, b_probs, actions, returns_from_rewards)

    print(f"target theta={theta}   ordinary IS mean={o_est:.3f}   se={o_se:.4f}")
    print(f"target theta={theta}   weighted IS mean={w_est:.3f}   se={w_se:.4f}")
