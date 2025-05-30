"""
Value Iteration:
- Input: Tabular MDP, discount factor, and initial value function.
- Output: Optimal value function and policy.

Algorithm outline:
1. Initialize the value function arbitrarily.
2. Repeat until convergence:
   a. For each state, compute the value function using the Bellman optimality equation.
3. Derive the optimal policy from the value function by greedy action selection.
"""

from rl_david_silver_2015.mdp.tabular_mdpf import TabularMDP
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp

TabularV = Float[Array, "n_states"]
TabularQ = Float[Array, "n_states n_actions"]

def bellman_optimality_v(mdp: TabularMDP, v0: TabularV) -> TabularV:
    """
    for each state, see the expected value by testing the value of all actions, 
    which needs to integrate over all possible states to transition to
    """
    state_action_values: Float[Array, "n_states n_actions"] = jnp.einsum('...d,d->...', mdp.P, v0) # dotting the each state probability vector with the value vector
    state_optimal_action_values: Float[Array, "n_states"] = jnp.max(state_action_values, axis=1)
    return state_optimal_action_values

def value_iteration_nonjit(mdp: TabularMDP, v0: TabularV, tolerance: float = 1e-6):
    last_improvement: Float[Array, ""] = jnp.array(jnp.inf)
    while last_improvement > tolerance:
        # apply bellman optimality equation
        v1 = bellman_optimality_v(mdp, v0)
        last_improvement = jnp.mean((v1-v0)**2, axis=1)
        v0 = v1
    return v0
