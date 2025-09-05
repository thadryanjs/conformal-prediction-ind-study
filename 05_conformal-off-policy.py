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
from typing import Callable, List, Tuple, Any
import numpy as np
import random

State = Any
Action = Any
Reward = float

@dataclass(frozen=True)
class SimulationResult:
    state: State
    action: Action
    reward: Reward
    next_state: State
    prob_pi_b: float = 0.0    # πb(at | xt) recorded at sampling time
    prob_pi: float = 0.0      # π(at | xt) for target policy (for weighting)

Trajectory = List[SimulationResult]


# %% [code]
@dataclass
class OffPolicySimulation:
    states: List[State]
    actions: List[Action]
    horizon: int
    reward_fn: Callable[[State, Action, State], Reward]
    # sampler for behavior policy πb (must be provided)
    policy_sampler_b: Callable[[State], Action]
    # probability fn for πb (must be provided)
    policy_prob_b: Callable[[State], np.ndarray]
    # probability fn for target π (must be provided)
    policy_prob_target: Callable[[State], np.ndarray]
    transition_fn: Callable[[State, Action], State]
    rng: np.random.Generator
    trajs: List[Trajectory] = field(default_factory=list)

    def simulate_traj(self, init_state: State = None) -> Trajectory:
        """
        Simulate one trajectory of length `horizon`.
        If init_state is None we sample a starting state uniformly from `states`.
        Requires policy_sampler_b, policy_prob_b and policy_prob_target to be provided.
        """
        traj: Trajectory = []
        # use numpy RNG for consistent reproducibility when needed
        state = init_state if init_state is not None else int(self.rng.choice(self.states))
        for t in range(self.horizon):
            # sample action using behavior policy sampler
            action = int(self.policy_sampler_b(state))
            # record probabilities under behavior and target policies
            p_b = float(self.policy_prob_b(state)[action])
            p_target = float(self.policy_prob_target(state)[action])
            next_state = self.transition_fn(state, action)
            reward = self.reward_fn(state, action, next_state)
            traj.append(SimulationResult(state, action, reward, next_state, prob_pi_b=p_b, prob_pi=p_target))
            state = next_state
        self.trajs.append(traj)
        return traj

    def sim_n_trajs(self, n: int, init_states: List[State] = None) -> List[Trajectory]:
        """Simulate n trajectories. Optionally provide a list of initial states."""
        results = []
        for i in range(n):
            init = None
            if init_states is not None and i < len(init_states):
                init = init_states[i]
            results.append(self.simulate_traj(init))
        return results

    def get_trajs(self, n: int = None) -> List[Trajectory]:
        return self.trajs if n is None else self.trajs[:n]


# %% [code]
# basic reward function
def reward_fn(state, action):
    return 0

# reward used in the paper
def reward_fn(
    current_inventory,
    order_quantity,
    next_inventory,
    max_capacity=10,
    fixed_order_cost=1.0,
    unit_purchase_cost=2.0,
    unit_holding_cost=2.0,
    unit_selling_price=4.0,
) -> float:
    """
    Reward r(current_inventory, order_quantity, next_inventory) = revenue - (fixed order cost + holding cost + purchase cost).
    """
    # Cast to floats for arithmetic
    current_inventory = float(current_inventory)
    order_quantity = float(order_quantity)
    next_inventory = float(next_inventory)

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
    return sales_revenue - (order_fee + holding_cost + purchase_cost)


# %% [code]
"""
We make a function that returns another function so we can use variables from the outer scope inside the class. This is slightly more complex, but means we don't have to have two different version of the same class if we want to compare two runs with different transition functions.
"""

# %% [code]
def build_transition_fn(max_capacity: int, demand_rate: float, rng: np.random.Generator = None) -> Callable[[int, int], int]:

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
transition_fn = build_transition_fn(max_capacity=10, demand_rate=5.0, rng=rng)
print([transition_fn(5, 5) for _ in range(25)])


# %% [code]
"""
this builds a random sampler where certain actions are favored because they are high-reward.
"""
def build_epsilon_greedy_policy(policy_hat: Callable[[State], Action],
                                n_actions: int,
                                epsilon: float,
                                rng: np.random.Generator = None
                               ) -> Tuple[Callable[[State], Action], Callable[[State], np.ndarray]]:
    rng = rng or np.random.default_rng()

    # this function assigns probabilities to each action accounting for the policy
    def action_prob_fn(state: State) -> np.ndarray:
        # this gives the same probabilities for all states (example: [0.05, 0.05])
        p = np.full(n_actions, epsilon / n_actions, dtype=float)
        # this establish the best action under the policy (example: 1)
        best = int(policy_hat(state))
        # add 1.0 - epsilon to the best action (example: [0.95, 0.05])
        p[best] += 1.0 - epsilon
        return p

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
sample_action_fn, action_prob_fn = build_epsilon_greedy_policy(policy_hat_fn, 2, 0.1, rng)

print(sample_action_fn(2), action_prob_fn(1))


# %% [code]
for i in range(10):
    print(sample_action_fn(i), action_prob_fn(i))


# %% [code]
"""
Make a policy sampler that favors the best action but allows some exploration
"""
def make_epsilon_greedy_factory(policy_hat: Callable[[State], Action],
                                n_actions: int,
                                epsilon: float,
                                rng: np.random.Generator):
    def action_prob_fn(state: State) -> np.ndarray:
        p = np.full(n_actions, epsilon / n_actions, dtype=float)
        best = int(policy_hat(state))
        p[best] += 1.0 - epsilon
        return p

    def sample_action_fn(state: State) -> int:
        p = action_prob_fn(state)
        return int(rng.choice(n_actions, p=p))

    return sample_action_fn, action_prob_fn

# example deterministic base policy
def example_policy_hat(state: int) -> int:
    threshold = 3
    return 1 if state < threshold else 0

# test it out
pi_b_sampler, pi_b_prob = make_epsilon_greedy_factory(example_policy_hat, n_actions=2, epsilon=0.4, rng=rng)
# we don't need the sampler for the this one because we're not generating trajs
_, pi_prob = make_epsilon_greedy_factory(example_policy_hat, n_actions=2, epsilon=0.2, rng=rng)


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
    rng=rng
)

trajs = sim.sim_n_trajs(5)
for i, tr in enumerate(trajs):
    print(f"Traj {i}: return={sum(s.reward for s in tr)}, initial_state={tr[0].state}")
    print(" per-step probs (πb, π):", [(s.prob_pi_b, s.prob_pi) for s in tr])
