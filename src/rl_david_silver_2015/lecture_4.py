"""
Goal in this lecture:
We have an unknown MDP, but we can sample from it. How do we learn its value function?
"""

"""
Approach 1: Monte Carlo Policy Evaluation (MCPE)
Variant 1: First-visit MCPE
Variant 2: Every-visit MCPE
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from rl_david_silver_2015.gridworld2d import create_gridworld2d
from rl_david_silver_2015.mdp import MDP, Policy, create_random_policy

# def mcpe_first_visit(mdp: MDP, policy: Policy, num_episodes=1000):
    # pass

def sample_mdp_one_step(mdp: MDP, policy: Policy, states: Int[Array, "batch"], b: int = 1) -> Tuple[Float[Array, "batch"], Float[Array, "batch"], Float[Array, "batch"]]:
    """
    Sample one step from the MDP given a state and policy.
    Returns the next state and reward.
    Allows batching by setting n > 1.
    """
    assert policy.shape[0] == mdp.n_states, f"Policy shape {policy.shape} does not match MDP shape {mdp.n_states}"
    assert policy.shape[1] == mdp.n_actions, f"Policy shape {policy.shape} does not match MDP shape {mdp.n_actions}"
    # Sample action from the policy
    actions = jax.random.categorical(jax.random.PRNGKey(None), policy[states], shape=(b,))

    # Sample next state from the transition probabilities
    next_states = jax.vmap(lambda s, a: jax.random.categorical(jax.random.PRNGKey(None), mdp.transition[s, a]))(states, actions)
    rewards = jax.vmap(lambda s, a: mdp.reward[s, a])(states, actions)

    return actions, next_states, rewards

def sample_mdp_n_step(mdp: MDP, policy: Policy, states: Int[Array, "batch"], b: int = 1, n: int = 1) -> Tuple[Float[Array, "steps batch"], Float[Array, "steps batch"], Float[Array, "steps batch"]]:
    """
    Sample n steps from the MDP given a batch of states and policy.
    Returns the actions, next states, and rewards for each step.
    Allows batching by setting b > 1 and multiple steps by setting n > 1.
    """
    assert policy.shape[0] == mdp.n_states, f"Policy shape {policy.shape} does not match MDP shape {mdp.n_states}"
    assert policy.shape[1] == mdp.n_actions, f"Policy shape {policy.shape} does not match MDP shape {mdp.n_actions}"

    actions = jnp.zeros((n, b), dtype=jnp.int32)
    next_states = jnp.zeros((n, b), dtype=jnp.int32)
    rewards = jnp.zeros((n, b), dtype=jnp.float32)

    for i in range(n):
        # Sample actions for each state in the batch
        actions = actions.at[i].set(jax.random.categorical(jax.random.PRNGKey(i), policy[states], shape=(b,)))

        # Sample next states and rewards for each state-action pair in the batch
        next_states = next_states.at[i].set(
            jax.vmap(lambda s, a: jax.random.categorical(jax.random.PRNGKey(i), mdp.transition[s, a]))(states, actions[i])
        )
        rewards = rewards.at[i].set(
            jax.vmap(lambda s, a: mdp.reward[s, a])(states, actions[i])
        )

        # Update states for the next step
        states = next_states[i]

    return actions, next_states, rewards

if __name__ == "__main__":
    # Example usage
    mdp = create_gridworld2d()
    policy = create_random_policy(mdp)
    a, s, r = sample_mdp_n_step(mdp, policy, states=jnp.array([5, 3, 2, 0, 7]), b=5, n=10)
    print("Actions:\n", a)
    print("Next States:\n", s)
    print("Rewards:\n", r)