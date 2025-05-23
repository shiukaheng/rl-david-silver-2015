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

def sample_mdp_n_step(
    mdp: MDP,
    policy: Policy,
    states: Int[Array, "batch"],
    b: int = 1,
    n: int = 1,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(42),
) -> Tuple[Float[Array, "steps batch"], Float[Array, "steps batch"], Float[Array, "steps batch"]]:
    """
    Sample n steps from the MDP given a batch of states and a policy.
    Returns (actions, next_states, rewards) arrays of shape (n, b).
    """
    assert policy.shape[0] == mdp.n_states, f"Policy shape {policy.shape} does not match MDP state count {mdp.n_states}"
    assert policy.shape[1] == mdp.n_actions, f"Policy shape {policy.shape} does not match MDP action count {mdp.n_actions}"

    actions = jnp.zeros((n, b), dtype=jnp.int32)
    next_states = jnp.zeros((n, b), dtype=jnp.int32)
    rewards = jnp.zeros((n, b), dtype=jnp.float32)

    for i in range(n):
        # Split RNG for this step
        rng, step_key = jax.random.split(rng)
        action_keys = jax.random.split(step_key, b)

        # Sample actions from the policy
        sampled_actions = jax.vmap(lambda key, s: jax.random.categorical(key, policy[s]))(action_keys, states)
        actions = actions.at[i].set(sampled_actions)

        # Split RNG again for transitions
        rng, step_key = jax.random.split(rng)
        trans_keys = jax.random.split(step_key, b)

        # Sample next states
        sampled_next_states = jax.vmap(lambda key, s_a: jax.random.categorical(key, mdp.transition[s_a[0], s_a[1]]))(
            trans_keys, jnp.stack([states, sampled_actions], axis=1)
        )
        next_states = next_states.at[i].set(sampled_next_states)

        # Lookup rewards
        sampled_rewards = jax.vmap(lambda s, a: mdp.reward[s, a])(states, sampled_actions)
        rewards = rewards.at[i].set(sampled_rewards)

        # Update state for next step
        states = sampled_next_states

    return actions, next_states, rewards

if __name__ == "__main__":
    # Example usage
    mdp = create_gridworld2d()
    policy = create_random_policy(mdp)
    a, s, r = sample_mdp_n_step(mdp, policy, states=jnp.array([5, 3, 2, 0, 7]), b=5, n=10)
    print("Actions:\n", a)
    print("Next States:\n", s)
    print("Rewards:\n", r)