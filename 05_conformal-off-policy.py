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
Trajectory = List[SimulationResult]


@dataclass(frozen=True)
class SimulationResult:
    state: State
    action: Action
    reward: Reward


@dataclass
class OffPolicySimulation:
    states: List[State]
    actions: List[Action]
    horizon: int
    # this syntax means: (state, action, state) -> reward
    reward_fn: Callable[[State, Action, State], Reward]
    policy_fn: Callable[[State], Action]
    transition_fn: Callable[[State, Action], State]
    # prevents weird python list/constructor behavior
    trajs: List[Trajectory] = field(default_factory=list)

    def simulate_traj(self, init_state: State = None) -> Trajectory:
        """
        Simulate one trajectory of length `horizon`.
        If init_state is None we sample a starting state uniformly from `states`.
        """
        traj: Trajectory = []
        state = init_state if init_state is not None else random.choice(self.states)
        for t in range(self.horizon):
            action = self.policy_fn(state)
            next_state = self.transition_fn(state, action)
            reward = self.reward_fn(state, action, next_state)
            traj.append(SimulationResult(state, action, reward))
            # If you have a transition function, update state here:
            # state = transition_fn(state, action)
            # For now we sample next state uniformly (or keep deterministic) as example:
            state = random.choice(self.states)
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
sim = OffPolicySimulation(
    states=[0, 1, 2],
    actions=[0, 1],
    horizon=10,
    reward_fn=reward_fn,
    policy_fn=policy_fn,
    transition_fn=transition_fn
)

sim.sim_n_trajs(10)
