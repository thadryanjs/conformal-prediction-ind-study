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
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Any, Union

import numpy as np
from numpy.typing import NDArray

import random

State = Any
Action = Any
Reward = np.float64

@dataclass(frozen=True)
class TrajectoryStep:
    state: State
    action: Action
    reward: Reward
    next_state: State
    prob_pi_b: NDArray[np.float64]
    prob_pi: NDArray[np.float64]

@dataclass(frozen=True)
class Trajectory:
    trajs: List[TrajectoryStep]
    cumulative_reward: np.float64

# %% [code]
@dataclass
class OffPolicySimulation:
    states: List[State]
    actions: List[Action]
    horizon: int
    reward_fn: Callable[[State, Action, State], Reward]
    policy_sampler_b: Callable[[State], int]
    policy_prob_b: Callable[[State], NDArray[np.float64]]
    policy_prob_target: Callable[[State], NDArray[np.float64]]
    transition_fn: Callable[[State, Action], State]
    rng: np.random.Generator
    trajs: List[Trajectory] = field(default_factory=list)

    def simulate_traj(self) -> Trajectory:
        """
        """
        traj_steps: List[TrajectoryStep] = []
        # use RNG for consistent reproducibility when needed
        state = self.rng.choice(self.states)
        cumulative_reward = np.float64(0.0)
        for t in range(self.horizon):
            # sample action using behavior policy sampler
            action = int(self.policy_sampler_b(state))
            # record probabilities under behavior and target policies
            p_b_arr = np.asarray(self.policy_prob_b(state), dtype=np.float64)
            p_target_arr = np.asarray(self.policy_prob_target(state), dtype=np.float64)
            p_b = p_b_arr[action]
            next_state = self.transition_fn(state, action)
            reward = self.reward_fn(state, action, next_state)
            traj_steps.append(
                TrajectoryStep(
                    state, action, reward, next_state,
                    prob_pi_b=p_b_arr,
                    prob_pi=p_target_arr,
                )
            )
            state = next_state
            cumulative_reward += np.float64(reward)
        return Trajectory(traj_steps, cumulative_reward)

    def sim_n_trajs(self, n: int) -> None:
        """Simulate n trajectories. Optionally provide a list of initial states."""
        results: List[Trajectory] = []
        for i in range(n):
            results.append(self.simulate_traj())
        self.trajs = results

    def get_trajs(self, n: Union[int, None] = 10) -> List[Trajectory]:
        return self.trajs if n is None else self.trajs[:n]


# %% [code]
# reward used in the paper
# TODO: types?
def reward_fn(
    current_inventory,
    order_quantity,
    next_inventory,
    max_capacity=10,
    fixed_order_cost=1.0,
    unit_purchase_cost=2.0,
    unit_holding_cost=2.0,
    unit_selling_price=4.0,
) -> np.float64:
    """
    Reward r(current_inventory, order_quantity, next_inventory) = revenue - (fixed order cost + holding cost + purchase cost).
    """
    # Cast to floats for arithmetic
    current_inventory = np.float64(current_inventory)
    order_quantity = np.float64(order_quantity)
    next_inventory = np.float64(next_inventory)

    order_fee = fixed_order_cost if order_quantity > 0 else 0.0
    # can't order more than max capacity
    post_order_inventory = min(max_capacity, current_inventory + order_quantity)
    # units bought is delta in before/after inventory
    units_bought = max(0.0, post_order_inventory - current_inventory)
    # cost of current inventory
    holding_cost = unit_holding_cost * current_inventory
    # cost for the units bought
    purchase_cost = unit_purchase_cost * units_bought
    # units sold is the delta from post-order inventory to next_inventory
    units_sold = max(0.0, post_order_inventory - next_inventory)
    # gross revenue
    sales_revenue = unit_selling_price * units_sold
    # revenue minus costs
    return np.float64(sales_revenue - (order_fee + holding_cost + purchase_cost))


# %% [code]
"""
We make a function that returns another function so we can use variables from the outer scope inside the class. This is slightly more complex, but means we don't have to have two different version of the same class if we want to compare two runs with different transition functions.
"""


# %% [code]
def build_transition_fn(
    max_capacity: int, demand_rate: np.float64, rng: np.random.Generator
) -> Callable[[int, int], int]:

    rng = rng or np.random.default_rng()

    def transition(state: int, action: int) -> int:
        # inventory after order capped by capacity
        post_order = min(max_capacity, state + action)
        # demand sampled from Poisson; ensure integer
        demand = int(rng.poisson(demand_rate))
        next_state = max(0, post_order - demand)
        return next_state

    return transition


# this will apply these values to anything we pass to it
rng = np.random.default_rng(0)
# rng should be in a differnt block for ipython


# %% [code]
# example:
transition_fn = build_transition_fn(max_capacity=10, demand_rate=np.float64(5.0), rng=rng)
print([transition_fn(5, 5) for _ in range(25)])


# %% [code]
"""
this builds a random sampler where certain actions are favored because they are high-reward.
"""
def build_epsilon_greedy_policy(
    policy_hat: Callable[[State], Action],
    n_actions: int,
    epsilon: np.float64,
    rng: np.random.Generator,
) -> Tuple[Callable[[State], int], Callable[[State], NDArray[np.float64]]]:
    # this function assigns probabilities to each action accounting for the policy
    def action_prob_fn(state: State) -> NDArray[np.float64]:
        # this gives the same probabilities for all states (example: [0.05, 0.05])
        p = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
        # this establish the best action under the policy (example: 1)
        best = int(policy_hat(state))
        # add 1.0 - epsilon to the best action (example: [0.95, 0.05])
        p[best] += 1.0 - epsilon
        return np.asarray(p, dtype=np.float64)

    # this does the actual sampling of the action
    def sample_action_fn(state: State) -> int:
        p = action_prob_fn(state)
        return int(rng.choice(n_actions, p=p))

    return sample_action_fn, action_prob_fn


# example deterministic base policy
def policy_hat_fn(state: int) -> int:
    threshold = 3
    return 1 if state < threshold else 0


# test it out
sample_action_fn, action_prob_fn = build_epsilon_greedy_policy(
    policy_hat_fn, 2, np.float64(0.1), rng
)

print(sample_action_fn(2), action_prob_fn(1))


# %% [code]
for i in range(10):
    print(sample_action_fn(i), action_prob_fn(i))


# %% [code]
# example deterministic base policy
def example_policy_hat(state: int) -> int:
    threshold = 3
    return 1 if state < threshold else 0


# test it out
pi_b_sampler, pi_b_prob = build_epsilon_greedy_policy(
    example_policy_hat, n_actions=2, epsilon=np.float64(0.4), rng=rng
)
# we don't need the sampler for the this one because we're not generating trajs
_, pi_prob = build_epsilon_greedy_policy(
    example_policy_hat, n_actions=2, epsilon=np.float64(0.2), rng=rng
)


# %% [code]
sim = OffPolicySimulation(
    states=list(range(0, 11)),
    actions=[0, 1],
    horizon=10,
    reward_fn=reward_fn,
    policy_sampler_b=pi_b_sampler,
    policy_prob_b=pi_b_prob,
    policy_prob_target=pi_prob,
    transition_fn=transition_fn,
    rng=rng,
)


# %% [code]
sim.sim_n_trajs(10)

trajs = sim.get_trajs(5)

for i, tr in enumerate(trajs):
    print(f"Trajectory {i}")


