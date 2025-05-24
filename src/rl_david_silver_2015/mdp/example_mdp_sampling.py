from rl_david_silver_2015.mdp.gridworld_2d import create_gridworld_2d_tabular
from rl_david_silver_2015.mdp.mdp_sampling import sample_mdp_batched_generator

import jax.numpy as jnp


if __name__ == "__main__":
    mdp, idx_to_state, state_to_idx = create_gridworld_2d_tabular()
    uniform_policy = mdp.create_uniform_policy()
    initial_states = [(0, 3), (3, 0)]
    initial_states = jnp.array([state_to_idx(s) for s in initial_states], dtype=jnp.int32)

    for update in sample_mdp_batched_generator(mdp, uniform_policy, initial_states):
        print(update)